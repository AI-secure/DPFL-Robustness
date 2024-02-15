import torch.nn as nn
import logging
logger = logging.getLogger("logger")


def clean_test(helper, epoch, model):
    model.eval()
    total_loss = 0
    correct = 0
    dataset_size = 0

    data_iterator = helper.test_data
    for batch_id, batch in enumerate(data_iterator):
        data, targets = helper.get_batch(data_iterator, batch, evaluation=True)
        dataset_size += len(data)
        output = model(data)

        total_loss += nn.functional.cross_entropy(output, targets,
                                                  reduction='sum').item()
        # get the index of the max log-probability
        pred = output.data.max(1)[1]
        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(dataset_size)
                   ) if dataset_size != 0 else 0
    total_l = total_loss / dataset_size if dataset_size != 0 else 0
    acc = round(acc, 4)
    total_l = round(total_l, 4)
    logger.info('___Test-clean, epoch: {}: loss: {:.4f}, '
                'Acc: {}/{} ({:.4f}%)'.format(epoch,
                                              total_l, correct, dataset_size,
                                              acc))

    model.train()
    return (total_l, acc, correct, dataset_size)


def poison_test(helper, epoch, model):
    model.eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count = 0

    data_iterator = helper.test_data_poison
    for batch_id, batch in enumerate(data_iterator):
        if helper.params['adv_method'] == 'labelflip':  # label -flipping
            data, targets, poison_num = helper.get_poison_batch(
                batch, adversarial_index=0, evaluation=True)
        elif helper.params['adv_method'] == 'backdoor':  # backdoor
            data, targets, poison_num = helper.get_poison_batch(
                batch, adversarial_index=1, evaluation=True)

        poison_data_count += poison_num
        dataset_size += len(data)
        output = model(data)

        total_loss += nn.functional.cross_entropy(output, targets,
                                                  reduction='sum').item()
        # get the index of the max log-probability
        pred = output.data.max(1)[1]
        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(poison_data_count)
                   ) if poison_data_count != 0 else 0
    total_l = total_loss / poison_data_count if poison_data_count != 0 else 0
    acc = round(acc, 4)
    total_l = round(total_l, 4)
    logger.info('___Test-poison , epoch: {}: loss: {:.4f}, '
                'Acc: {}/{} ({:.4f}%)'.format(epoch,
                                              total_l, correct, poison_data_count,
                                              acc))
    model.train()
    return total_l, acc, correct, poison_data_count
