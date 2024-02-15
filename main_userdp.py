from helper import *
import copy
from local_train import FLTrain_UserDPFL
from opacus import PrivacyEngine
import config
import random
import numpy as np
import time
import yaml
import utils.csv_record as csv_record
from image_helper import ImageHelper
import test
import torch
import logging
import datetime
import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
logger = logging.getLogger("logger")


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="mnist_params_ceracc.yaml")
    parser.add_argument(
        "--is_poison",
        action="store_true",
        help='perform attacks'
    )

    parser.add_argument("--num_adv", type=int, default=0)
    parser.add_argument("--adv_method", type=str, default='backdoor',
                        choices=['labelflip', 'backdoor']
                        )
    parser.add_argument("--n_runs", type=int, default=1,
                        help='number of runs for Monte Carlo Approximation'
                        )

    parser.add_argument("--scale_factor", type=int, default=0)
    parser.add_argument("--fl_aggregation", type=str, default="fedavg",
                        choices=['fedavg', 'rfa', 'krum', 'mkrum',
                                 'bulyan', 'median', 'trmean'],
                        help='Aggregation method'
                        )
    parser.add_argument("--pre_path", type=str, default="saved_models")
    parser.add_argument("--dpfl", type=str, default="max_model",
                        choices=['median_per_layer', 'max_per_layer',
                                 'max_model', 'median_model'],
                        help="User-level DPFL mechanism for fedavg: a fixed max value v.s. medium norm as clipping threshold; layer norm v.s. whole model norm clipping")

    args = parser.parse_args()

    print(args)

    with open(args.config, 'r') as f:
        params_loaded = yaml.load(f, Loader=yaml.FullLoader)

    dataset = params_loaded['type']
    if args.is_poison:
        args.pre_path = f"saved_models/{dataset}_userdp_{args.adv_method}_adv{args.num_adv}"
    else:
        args.pre_path = f"saved_models/{dataset}_userdp"
    args_dict = vars(args)
    params_loaded.update(args_dict)

    if params_loaded['is_poison'] == True:  # if performing poisoning attack
        params_loaded['adversary_list'] = list(
            range(1, params_loaded['num_adv'] + 1))
        # if performing distributed poisoning attacks, split the backdoor pixels
        if params_loaded['dba'] == True:
            pattern = params_loaded['poison_pattern']
            per_pixel = int(len(pattern)/len(params_loaded['adversary_list']))
            print(per_pixel)
            for i in range(len(params_loaded['adversary_list'])-1):
                adv_name = params_loaded['adversary_list'][i]
                params_loaded[str(
                    adv_name)+'_poison_pattern'] = pattern[i*per_pixel:per_pixel*(i+1)]
                print(str(adv_name)+'_poison_pattern',
                      params_loaded[str(adv_name)+'_poison_pattern'])
            i = len(params_loaded['adversary_list'])-1
            adv_name = params_loaded['adversary_list'][i]
            params_loaded[str(adv_name) +
                          '_poison_pattern'] = pattern[i*per_pixel:-1]
            print(str(adv_name)+'_poison_pattern',
                  params_loaded[str(adv_name)+'_poison_pattern'])

    set_random_seed(0)  # fix the seed for create local datasets

    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')

    if params_loaded['type'] == config.TYPE_CIFAR or params_loaded['type'] == config.TYPE_MNIST:
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', params_loaded['type']))
        helper.load_data()
    else:
        helper = None
        logger.info(f'Datasets are not supported')
        exit(0)

    logger.info(f'load data done')

    for run_idx in range(0, params_loaded['n_runs']):
        logger.info(f'start run number:{run_idx}')

        torch.cuda.empty_cache()
        set_random_seed(run_idx)  # set the pre-defined seed for dp randomness
        helper.create_model()
        logger.info(f'create model done')

        if params_loaded['withDP'] == True:
            g_optimizer = torch.optim.SGD(
                helper.target_model.parameters(), lr=helper.params['lr'])

            global_privacy_engine = PrivacyEngine(
                helper.target_model,
                batch_size=params_loaded['no_models'],  # selected clients num
                # total number of clients
                sample_size=params_loaded['number_of_total_participants'],
                alphas=[
                    1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier=params_loaded['noise_multiplier'],
                max_grad_norm=params_loaded['max_clip_norm'])  # for weight norm

            global_privacy_engine.attach(g_optimizer)

        # Create models
        if helper.params['is_poison']:
            logger.info(
                f"Poison following participants: {(helper.params['adversary_list'])}")

        # save parameters:
        with open(f'{helper.folder_path}/params.yaml', 'w') as f:
            yaml.dump(helper.params, f)

        for epoch in range(helper.start_epoch, helper.params['epochs'] + 1):

            agent_name_keys = np.random.choice(range(params_loaded['number_of_total_participants']),
                                               max(params_loaded['no_models'], 1),
                                               replace=False)
            n_attacker_round = 0
            for agent_name_key in agent_name_keys:
                if helper.params['is_poison'] and agent_name_key in helper.params['adversary_list']:
                    n_attacker_round += 1

            start_time = time.time()

            submit_params_update_dict, num_samples_dict, num_poisoned_samples_dict = FLTrain_UserDPFL(
                helper=helper,
                logger=logger,
                start_epoch=epoch,
                local_model=helper.local_model,
                target_model=helper.target_model,
                is_poison=helper.params['is_poison'],
                agent_name_keys=agent_name_keys)

            if helper.params['fl_aggregation'] == 'fedavg':
                logger.info(f"DPFL methods: {helper.params['dpfl']}")
                if helper.params['dpfl'] == 'median_model':
                    clip_norm = helper.compute_median_norm(
                        submit_params_update_dict, agent_name_keys)
                    helper.fedavg_clientdp(submit_params_update_dict,
                                           agent_name_keys,
                                           clip_norm=clip_norm,
                                           target_model=helper.target_model)
                elif helper.params['dpfl'] == 'median_per_layer':
                    clip_norm = helper.compute_median_norm_per_layer(
                        submit_params_update_dict, agent_name_keys)
                    helper.fedavg_clientdp_per_layer(submit_params_update_dict,
                                                     agent_name_keys,
                                                     layers_clip_norm=clip_norm,
                                                     target_model=helper.target_model)
                elif helper.params['dpfl'] == 'max_per_layer':
                    clip_norm = helper.set_max_norm_per_layer(
                        submit_params_update_dict, agent_name_keys, helper.params['max_clip_norm'])

                    helper.fedavg_clientdp_per_layer(submit_params_update_dict,
                                                     agent_name_keys,
                                                     layers_clip_norm=clip_norm,
                                                     target_model=helper.target_model)
                else:
                    clip_norm = helper.params['max_clip_norm']
                    helper.fedavg_clientdp(submit_params_update_dict,
                                           agent_name_keys,
                                           clip_norm=clip_norm,
                                           target_model=helper.target_model)
            else:

                if helper.params['fl_aggregation'] == 'rfa':
                    maxiter = helper.params['geom_median_maxiter']
                    updates = dict()
                    for name_key in agent_name_keys:
                        updates[name_key] = (num_samples_dict[name_key], copy.deepcopy(
                            submit_params_update_dict[name_key]))
                    num_oracle_calls, is_updated, names, weights, alphas = helper.geometric_median_update(
                        helper.target_model, updates, maxiter=maxiter)
                else:
                    user_grads = []
                    for agent_name_key in agent_name_keys:
                        param_grad = flatten(
                            submit_params_update_dict[agent_name_key])
                        user_grads = param_grad[None, :] if len(user_grads) == 0 else torch.cat(
                            (user_grads, param_grad[None, :]), 0)

                    if helper.params['fl_aggregation'] == 'krum' or helper.params['fl_aggregation'] == 'mkrum':
                        multi_k = True if helper.params['fl_aggregation'] == 'mkrum' else False
                        agg_grads, krum_candidate = multi_krum(
                            user_grads, n_attacker_round, multi_k=multi_k)
                    elif helper.params['fl_aggregation'] == 'bulyan':
                        agg_grads, krum_candidate = bulyan(
                            user_grads, n_attacker_round)

                    elif helper.params['fl_aggregation'] == 'median':
                        agg_grads = torch.median(user_grads, dim=0)[0]

                    elif helper.params['fl_aggregation'] == 'trmean':
                        agg_grads = tr_mean(user_grads, n_attacker_round)

                    agg_params_update = unflatten(
                        agg_grads, submit_params_update_dict[agent_name_keys[0]])
                    for name, layer in helper.target_model.state_dict().items():
                        if 'num_batches_tracked' in name:
                            continue
                        layer.add_(agg_params_update[name])

            epoch_loss, epoch_acc, epoch_corret, epoch_total = test.clean_test(helper=helper, epoch=epoch,
                                                                               model=helper.target_model)
            p_epoch_loss = 0
            epoch_acc_p = 0
            if params_loaded['is_poison'] == True:
                p_epoch_loss, epoch_acc_p, epoch_corret, epoch_total = test.poison_test(helper=helper,
                                                                                        epoch=epoch,
                                                                                        model=helper.target_model)
                if helper.params['record_p'] == True:
                    csv_record.posiontest_result.append(
                        ["global", epoch, p_epoch_loss, epoch_acc_p])

            if params_loaded['withDP'] == True:
                global_privacy_engine.steps = epoch  # assign the epoch
                epsilon, best_alpha = global_privacy_engine.get_privacy_spent(
                    params_loaded['delta'])
                epsilon = round(epsilon, 4)  # 4 digit
                logger.info('___GlobalDP, epoch: {},  accuracy: {:.4f} epsilon: {:.4f}, clip norm: {:.4f}, noise_mul:{} delta: {} for alpha: {}'
                            .format(epoch, epoch_acc, epsilon, clip_norm,  params_loaded['noise_multiplier'],  params_loaded['delta'], best_alpha))
                csv_record.dp_result.append([epoch,  epsilon, epoch_acc, epoch_loss,
                                             p_epoch_loss, epoch_acc_p])
            else:
                csv_record.dp_result.append([epoch, 0.0, epoch_acc, epoch_loss,
                                             p_epoch_loss, epoch_acc_p])
            if epoch == helper.start_epoch:
                logger.info(
                    f'Done one epoch in {time.time() - start_time} sec.')

            helper.save_model_for_certify(epoch=epoch, run_idx=run_idx)
            csv_record.save_result_csv(helper.folder_path, run_idx=run_idx)

        csv_record.clear_csv()
