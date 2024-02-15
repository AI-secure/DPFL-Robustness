import test
from local_train import FLTrain_InsDPFL
from opacus import PrivacyEngine
import config
import random
import numpy as np
import time
import yaml
import utils.csv_record as csv_record
from image_helper import ImageHelper
import torch
import logging
import datetime
import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
logger = logging.getLogger("logger")


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_global_eps(delta, local_privacy_engines, agent_names):
    max_eps = 0
    alpha = 0

    for ag_name in agent_names:
        epsilon, best_alpha = local_privacy_engines[ag_name].get_privacy_spent(
            delta)
        if epsilon > max_eps:
            max_eps = epsilon
            alpha = best_alpha
    return max_eps, alpha


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="mnist_params_insdp_ceracc.yaml")
    parser.add_argument(
        "--is_poison",
        action="store_true",
        help='perform attacks'
    )
    parser.add_argument("--pre_path", type=str, default="saved_models")
    parser.add_argument("--num_adv", type=int, default=0)
    parser.add_argument("--adv_method", type=str, default='backdoor',
                        choices=['labelflip', 'backdoor']
                        )
    parser.add_argument("--n_runs", type=int, default=1,
                        help='number of runs for Monte Carlo Approximation'
                        )

    args = parser.parse_args()
    print(args)

    with open(args.config, 'r') as f:
        params_loaded = yaml.load(f, Loader=yaml.FullLoader)

    dataset = params_loaded['type']
    if args.is_poison:
        args.pre_path = f"saved_models/{dataset}_insdp_{args.adv_method}_adv{args.num_adv}"
    else:
        args.pre_path = f"saved_models/{dataset}_insdp"
    
    args_dict = vars(args)
    params_loaded.update(args_dict)

    if params_loaded['is_poison'] == True:
        params_loaded['adversary_list'] = list(
            range(1, params_loaded['num_adv'] + 1))
        if params_loaded['dba'] == True and params_loaded['is_poison'] == True:
            pattern = params_loaded['poison_pattern']
            per_pixel = int(len(pattern)/len(params_loaded['adversary_list']))
            print(per_pixel)
            for i in range(len(params_loaded['adversary_list'])):
                adv_name = params_loaded['adversary_list'][i]
                params_loaded[str(
                    adv_name)+'_poison_pattern'] = pattern[i*per_pixel:per_pixel*(i+1)]
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

    # save parameters:
    with open(f'{helper.folder_path}/params.yaml', 'w') as f:
        yaml.dump(helper.params, f)

    for run_idx in range(0, params_loaded['n_runs']):
        logger.info(f'start run number:{run_idx}')

        torch.cuda.empty_cache()
        set_random_seed(run_idx)  # set the pre-defined seed for dp randomness
        helper.create_model()
        logger.info(f'create model done')

        # init local
        local_privacy_engines = dict()
        local_optimizers = dict()
        local_models = dict()
        for agent_name in range(params_loaded['number_of_total_participants']):
            local_model = helper.create_one_model()
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, local_model.parameters()), lr=helper.params['lr'],
                                        momentum=helper.params['momentum'],
                                        weight_decay=helper.params['decay'])

            if params_loaded['withDP'] == True:
                privacy_engine = PrivacyEngine(
                    local_model,
                    batch_size=params_loaded['batch_size'],
                    sample_size=int(len(helper.train_dataset) /
                                    params_loaded['number_of_total_participants']),
                    alphas=[
                        1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                    noise_multiplier=helper.params['noise_multiplier'],
                    max_grad_norm=helper.params['max_clip_norm'],
                )
                privacy_engine.attach(optimizer)
                local_privacy_engines[agent_name] = privacy_engine
            local_optimizers[agent_name] = optimizer
            local_models[agent_name] = local_model

        for epoch in range(helper.start_epoch, helper.params['epochs'] + 1):

            agent_name_keys = np.random.choice(range(params_loaded['number_of_total_participants']),
                                               max(params_loaded['no_models'], 1),
                                               replace=False)
            start_time = time.time()

            submit_params_update_dict = FLTrain_InsDPFL(
                helper=helper,
                logger=logger,
                start_epoch=epoch,
                local_models=local_models,
                local_optimizers=local_optimizers,
                local_privacy_engines=local_privacy_engines,
                target_model=helper.target_model,
                is_poison=helper.params['is_poison'],
                agent_name_keys=agent_name_keys)

            helper.average_models_params(submit_params_update_dict,
                                         agent_name_keys,
                                         target_model=helper.target_model)

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
                epsilon, best_alpha = get_global_eps(params_loaded['delta'], local_privacy_engines, range(
                    params_loaded['number_of_total_participants']))

                epsilon = round(epsilon, 4)  # 4 digit
                logger.info('___GlobalDP, epoch: {},  accuracy: {:.4f} epsilon: {:.4f}, clip norm: {:.4f}, noise_mul:{} delta: {} for alpha: {}'
                            .format(epoch, epoch_acc, epsilon, helper.params['max_clip_norm'],
                                    params_loaded['noise_multiplier'],  params_loaded['delta'], best_alpha))
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

    logger.info(
        f"Done. This run has a label: {helper.params['current_time']}. ")
