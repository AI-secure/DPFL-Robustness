import torch
import torch.nn as nn
import test
from utils.utils import model_copy_params
import utils.csv_record as csv_record
import math


def model_dist_norm(model, target_params, _type):
    squared_sum = 0

    for name, layer in model.state_dict().items():
        if 'num_batches_tracked' in name:
            continue
        squared_sum += torch.sum(torch.pow(layer.data -
                                 target_params[name].data, 2))
    return math.sqrt(squared_sum)


def FLTrain_UserDPFL(helper, logger, start_epoch, local_model, target_model, is_poison, agent_name_keys):
    submit_params_update_dict = dict()
    num_samples_dict = dict()
    num_poisoned_samples_dict = dict()
    target_params = dict()
    for name, param in target_model.state_dict().items():
        if 'num_batches_tracked' in name:
            continue
        target_params[name] = target_model.state_dict(
        )[name].clone().detach().requires_grad_(False)

    for model_id in range(helper.params['no_models']):
        ADV_CLIENT = False

        agent_name_key = agent_name_keys[model_id]
        if is_poison and agent_name_key in helper.params['adversary_list']:
            ADV_CLIENT = True

        # Synchronize LR and models
        model = local_model
        model_copy_params(model, target_model.state_dict())
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=helper.params['lr'],
                                    momentum=helper.params['momentum'],
                                    weight_decay=helper.params['decay'])

        criterion = nn.CrossEntropyLoss().cuda()
        model.train()
        epoch = start_epoch
        temp_local_epoch = (epoch - 1) * helper.params['internal_epochs']

        for internal_epoch in range(1, helper.params['internal_epochs'] + 1):
            temp_local_epoch += 1
            _, data_iterator = helper.train_data[agent_name_key]
            total_loss = 0.
            correct = 0
            dataset_size = 0
            poison_data_count = 0
            dis2global_list = []
            model.train()
            for batch_id, batch in enumerate(data_iterator):

                optimizer.zero_grad()

                # label -flipping
                if ADV_CLIENT == True and helper.params['adv_method'] == 'labelflip':
                    data, targets, poison_num = helper.get_poison_batch(
                        batch, adversarial_index=0, evaluation=False)
                    poison_data_count += poison_num
                # backdoor
                elif ADV_CLIENT == True and helper.params['adv_method'] == 'backdoor':
                    if helper.params['dba'] == True:
                        data, targets, poison_num = helper.get_poison_batch(
                            batch, adversarial_index=1, evaluation=False, agent_name=agent_name_key)
                    else:
                        data, targets, poison_num = helper.get_poison_batch(
                            batch, adversarial_index=1, evaluation=False)
                    poison_data_count += poison_num
                else:
                    data, targets = helper.get_batch(
                        data_iterator, batch, evaluation=False)

                dataset_size += len(data)
                output = model(data)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.data
                # get the index of the max log-probability
                pred = output.data.max(1)[1]
                correct += pred.eq(targets.data.view_as(pred)
                                   ).cpu().sum().item()

            acc = 100.0 * (float(correct) / float(dataset_size))
            total_l = total_loss / dataset_size

            if internal_epoch == helper.params['internal_epochs'] and helper.params['record_local_train'] == True:
                if ADV_CLIENT == True:
                    logger.info(
                        '___Train Local adv,  epoch {:3d}, local model {}, local_epoch {:3d}, loss: {:.4f}, '
                        'Acc: {}/{} ({:.4f}%), num_adv_data: {}, dis: {:.3f}'.format(epoch, agent_name_key,
                                                                                     internal_epoch,
                                                                                     total_l, correct, dataset_size,
                                                                                     acc, poison_data_count,
                                                                                     model_dist_norm(model, target_params, helper.params['type'])))
                else:
                    logger.info(
                        '___Train Local, epoch {:3d}, local model {}, local_epoch {:3d}, loss: {:.4f}, '
                        'Acc: {}/{} ({:.4f}%) dis: {:.3f}'.format(epoch, agent_name_key, internal_epoch,
                                                                  total_l, correct, dataset_size,
                                                                  acc, model_dist_norm(model, target_params, helper.params['type'])))

            num_samples_dict[agent_name_key] = dataset_size
            num_poisoned_samples_dict[agent_name_key] = poison_data_count

        if ADV_CLIENT == True:
            for name, layer in model.state_dict().items():
                if 'num_batches_tracked' in name:
                    continue
                data = model.state_dict()[name]
                new_value = target_params[name] + \
                    (data - target_params[name]) * \
                    helper.params['scale_factor']
                model.state_dict()[name].copy_(new_value)
            if helper.params['record_p'] == True:
                epoch_loss, epoch_acc_p, epoch_corret, epoch_total = test.poison_test(helper=helper,
                                                                                      epoch=epoch,
                                                                                      model=model,
                                                                                      is_poison=ADV_CLIENT,
                                                                                      visualize=True,
                                                                                      agent_name_key=agent_name_key)

                csv_record.posiontest_result.append(
                    [agent_name_key, epoch, epoch_loss, epoch_acc_p])

        # update the model params
        client_pramas_update = dict()
        for name, data in model.state_dict().items():
            if 'num_batches_tracked' in name:
                continue

            client_pramas_update[name] = torch.zeros_like(data)
            client_pramas_update[name] = (
                data - target_model.state_dict()[name])

        submit_params_update_dict[agent_name_key] = client_pramas_update

    return submit_params_update_dict, num_samples_dict, num_poisoned_samples_dict


def FLTrain_InsDPFL(helper, logger, start_epoch, local_models, local_optimizers, local_privacy_engines, target_model, is_poison, agent_name_keys):

    submit_params_update_dict = dict()

    target_params = dict()
    for name, param in target_model.named_parameters():
        target_params[name] = target_model.state_dict(
        )[name].clone().detach().requires_grad_(False)

    for model_id in range(helper.params['no_models']):
        # client_grad = []
        agent_name_key = agent_name_keys[model_id]
        ADV_CLIENT = False
        if is_poison and agent_name_key in helper.params['adversary_list']:
            ADV_CLIENT = True

        # Synchronize LR and models
        model = local_models[agent_name_key]

        optimizer = local_optimizers[agent_name_key]
        if helper.params['withDP'] == True:
            local_privacy_engine = local_privacy_engines[agent_name_key]

        model_copy_params(model, target_model.state_dict())

        criterion = nn.CrossEntropyLoss().cuda()
        model.train()
        epoch = start_epoch
        temp_local_epoch = (epoch - 1) * helper.params['internal_epochs']

        for internal_epoch in range(1, helper.params['internal_epochs'] + 1):
            temp_local_epoch += 1

            _, data_iterator = helper.train_data[agent_name_key]
            total_loss = 0.
            correct = 0
            dataset_size = 0
            poison_data_count = 0
            optimizer.zero_grad()

            model.train()

            for batch_id, batch in enumerate(data_iterator):
                if ADV_CLIENT == True:
                    if batch_id == 0:
                        if helper.params['adv_method'] == 'labelflip':  # label -flipping
                            data, targets, poison_num = helper.get_poison_batch(
                                batch, adversarial_index=0, evaluation=False)
                            poison_data_count += poison_num
                        elif helper.params['adv_method'] == 'backdoor':  # backdoor
                            if helper.params['dba'] == True:
                                data, targets, poison_num = helper.get_poison_batch(
                                    batch, adversarial_index=1, evaluation=False, agent_name=agent_name_key)
                            else:
                                data, targets, poison_num = helper.get_poison_batch(
                                    batch, adversarial_index=1, evaluation=False)
                            poison_data_count += poison_num
                else:
                    data, targets = helper.get_batch(
                        data_iterator, batch, evaluation=False)

                dataset_size += len(data)
                output = model(data)
                loss = criterion(output, targets)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.data
                # get the index of the max log-probability
                pred = output.data.max(1)[1]
                correct += pred.eq(targets.data.view_as(pred)
                                   ).cpu().sum().item()

            acc = 100.0 * (float(correct) / float(dataset_size))
            total_l = total_loss / dataset_size

            if helper.params['record_local_train'] == True:
                logger.info(
                    '___Train Local, epoch {:3d}, local model {}, local_epoch {:3d}, Avg loss: {:.4f}, '
                    'Acc: {}/{} ({:.4f}%) '.format(epoch, agent_name_key, internal_epoch,
                                                   total_l, correct, dataset_size,
                                                   acc))

        # update the model params
        client_pramas_update = dict()
        for name, layer in model.named_parameters():
            data = model.state_dict()[name]
            client_pramas_update[name] = torch.zeros_like(data)
            client_pramas_update[name].add_(
                data - target_model.state_dict()[name])

        submit_params_update_dict[agent_name_key] = client_pramas_update

    return submit_params_update_dict
