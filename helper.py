import copy
import config
import numpy as np
import os
from shutil import copyfile
import math
import torch
import logging
logger = logging.getLogger("logger")


def flatten(weights_update):
    param_grad = []
    for name, data in weights_update.items():
        param_grad = data.view(-1) if not len(
            param_grad) else torch.cat((param_grad, data.view(-1)))
    return param_grad


def unflatten(flattened, normal_shape):
    weights_update = {}

    for name, data in normal_shape.items():
        n_params = len(data.view(-1))
        weights_update[name] = torch.as_tensor(
            flattened[:n_params]).reshape(data.size())
        flattened = flattened[n_params:]

    return weights_update


def multi_krum(all_updates, n_attackers, multi_k=False):
    nusers = all_updates.shape[0]
    candidates = []
    candidate_indices = []
    remaining_updates = all_updates
    all_indices = np.arange(len(all_updates))

    while len(remaining_updates) > 2 * n_attackers + 2:
        distances = []
        for update in remaining_updates:
            distance = torch.norm((remaining_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(
                distances) else torch.cat((distances, distance[None, :]), 0)

        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(
            distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        # [:len(remaining_updates) - 2 - n_attackers]
        indices = torch.argsort(scores)

        # if verbose: print(indices)
        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        candidates = remaining_updates[indices[0]][None, :] if not len(
            candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat(
            (remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
        if not multi_k:
            break

    aggregate = torch.mean(candidates, dim=0)
    return aggregate, np.array(candidate_indices)


def bulyan(all_updates, n_attackers):
    nusers = all_updates.shape[0]
    bulyan_cluster = []
    candidate_indices = []
    remaining_updates = all_updates
    all_indices = np.arange(len(all_updates))

    while len(bulyan_cluster) < (nusers - 2 * n_attackers):
        distances = []
        for update in remaining_updates:
            distance = torch.norm((remaining_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(
                distances) else torch.cat((distances, distance[None, :]), 0)

        distances = torch.sort(distances, dim=1)[0]

        scores = torch.sum(
            distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        indices = torch.argsort(scores)[:len(
            remaining_updates) - 2 - n_attackers]
        # logger.info(f'scores {scores}')
        # logger.info(f'indices {indices}')
        if len(indices) == 0:
            break
        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        bulyan_cluster = remaining_updates[indices[0]][None, :] if not len(
            bulyan_cluster) else torch.cat((bulyan_cluster, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat(
            (remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)

    # print('dim of bulyan cluster ', bulyan_cluster.shape)

    n, d = bulyan_cluster.shape
    param_med = torch.median(bulyan_cluster, dim=0)[0]
    sort_idx = torch.argsort(torch.abs(bulyan_cluster - param_med), dim=0)
    sorted_params = bulyan_cluster[sort_idx, torch.arange(d)[None, :]]

    return torch.mean(sorted_params[:n - 2 * n_attackers], dim=0), np.array(candidate_indices)


def tr_mean(all_updates, n_attackers):
    sorted_updates = torch.sort(all_updates, 0)[0]
    out = torch.mean(sorted_updates[n_attackers:-n_attackers],
                     0) if n_attackers else torch.mean(sorted_updates, 0)
    return out


class Helper:
    def __init__(self, current_time, params, name):
        self.current_time = current_time
        self.target_model = None
        self.local_model = None

        self.train_data = None
        self.test_data = None
        self.poisoned_data = None
        self.test_data_poison = None

        self.params = params
        self.name = name
        self.best_loss = math.inf
        self.pre_path = self.params['pre_path']
        try:
            os.mkdir(self.pre_path)
        except FileExistsError:
            logger.info('Folder already exists')

        self.folder_path = f'{self.pre_path}/model_{self.name}_{current_time}'
        try:
            os.mkdir(self.folder_path)
        except FileExistsError:
            logger.info('Folder already exists')
        logger.addHandler(logging.FileHandler(
            filename=f'{self.folder_path}/log.txt'))
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)
        logger.info(f'current path: {self.folder_path}')
        if not self.params.get('environment_name', False):
            self.params['environment_name'] = self.name

        self.params['current_time'] = self.current_time
        self.params['folder_path'] = self.folder_path

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if not self.params['save_model']:
            return False
        torch.save(state, filename)

        if is_best:
            copyfile(filename, 'model_best.pth.tar')

    @staticmethod
    def compute_update_norm(weights_update):
        squared_sum = 0
        for name, data in weights_update.items():
            squared_sum += torch.sum(torch.pow(data, 2))
        return math.sqrt(squared_sum)

    @staticmethod
    def clip_update_norm(weights_update, max_norm):
        total_norm = Helper.compute_update_norm(weights_update)
        clip_coef = max_norm / (total_norm + 1e-6)
        if total_norm > max_norm:
            for name, data in weights_update.items():
                data.mul_(clip_coef)

    def compute_median_norm(self, submit_params_update_dict, agent_name_keys):

        local_norms = []
        for i in range(0, len(agent_name_keys)):
            local_update = submit_params_update_dict[agent_name_keys[i]]
            local_norms.append(self.compute_update_norm(local_update))

        median_norm = np.median(local_norms)

        return median_norm

    def compute_median_norm_per_layer(self, submit_params_update_dict, agent_name_keys):
        layers_median_norm = dict()
        first_update = submit_params_update_dict[agent_name_keys[0]]

        for name, _ in first_update.items():
            if 'num_batches_tracked' in name:
                continue
            norms_all_clients = []
            for i in range(0, len(agent_name_keys)):
                local_update = submit_params_update_dict[agent_name_keys[i]]
                norms_all_clients.append(
                    math.sqrt(torch.sum(torch.pow(local_update[name], 2))))
            layers_median_norm[name] = np.median(norms_all_clients)

        return layers_median_norm

    def set_max_norm_per_layer(self, submit_params_update_dict, agent_name_keys, max_norm):
        layers_norm = dict()
        first_update = submit_params_update_dict[agent_name_keys[0]]
        for name, _ in first_update.items():
            layers_norm[name] = max_norm
        return layers_norm

    def fedavg_clientdp_per_layer(self, submit_params_update_dict, agent_name_keys, layers_clip_norm, target_model):
        """
        Perform FedAvg algorithm on model params

        """
        # clip
        if self.params['withDP'] == True:
            for i in range(0, len(agent_name_keys)):
                local_update = submit_params_update_dict[agent_name_keys[i]]
                self.clip_update_norm_per_layer(local_update, layers_clip_norm)

        # init the data structure
        agg_params_update = dict()
        for name, data in target_model.state_dict().items():
            if 'num_batches_tracked' in name:
                continue
            agg_params_update[name] = torch.zeros_like(data)
        # avg
        for name, data in agg_params_update.items():
            # avg
            for i in range(0, len(agent_name_keys)):
                client_params_update = submit_params_update_dict[agent_name_keys[i]]
                temp = client_params_update[name]
                data.add_(temp)
            # add noise
            if self.params['withDP'] == True:
                noise = torch.cuda.FloatTensor(data.shape).normal_(
                    mean=0, std=layers_clip_norm[name] * self.params['noise_multiplier'])
                data.add_(noise)
        for name, layer in target_model.state_dict().items():
            if 'num_batches_tracked' in name:
                continue
            layer.add_(agg_params_update[name] * 1.0/len(agent_name_keys))

    def fedavg_clientdp(self, submit_params_update_dict, agent_name_keys, clip_norm, target_model):
        """
        Perform FedAvg algorithm on model params

        """

        # clip
        if self.params['withDP'] == True:
            for i in range(0, len(agent_name_keys)):
                local_update = submit_params_update_dict[agent_name_keys[i]]
                self.clip_update_norm(local_update, clip_norm)

        # init the data structure
        agg_params_update = dict()
        for name, data in target_model.state_dict().items():
            if 'num_batches_tracked' in name:
                continue
            agg_params_update[name] = torch.zeros_like(data)
        # avg
        for name, data in agg_params_update.items():
            # avg
            for i in range(0, len(agent_name_keys)):
                client_params_update = submit_params_update_dict[agent_name_keys[i]]
                temp = client_params_update[name]
                data.add_(temp)
            # add noise
            if self.params['withDP'] == True:
                noise = torch.cuda.FloatTensor(data.shape).normal_(
                    mean=0, std=clip_norm * self.params['noise_multiplier'])
                data.add_(noise)
        for name, layer in target_model.state_dict().items():
            if 'num_batches_tracked' in name:
                continue
            layer.add_(agg_params_update[name] * 1.0/len(agent_name_keys))

    def average_models_params(self, submit_params_update_dict, agent_name_keys, target_model):
        """
        Perform FedAvg algorithm on model params

        """
        # init the data structure
        agg_params_update = dict()

        for name, data in target_model.state_dict().items():
            if 'num_batches_tracked' in name:
                continue
            agg_params_update[name] = torch.zeros_like(data)

        # avg
        for name, data in agg_params_update.items():
            # avg
            for i in range(0, len(agent_name_keys)):
                client_params_update = submit_params_update_dict[agent_name_keys[i]]
                temp = client_params_update[name]
                data.add_(temp)

        for name, layer in target_model.state_dict().items():
            if 'num_batches_tracked' in name:
                continue
            layer.add_(agg_params_update[name] * 1.0/len(agent_name_keys))

    def save_model_for_certify(self, model=None, epoch=0, run_idx=0):
        if model is None:
            model = self.target_model
        if self.params['save_model']:
            model_name = '{0}/model.pt.tar'.format(self.params['folder_path'])
            saved_dict = {'state_dict': model.state_dict(), 'epoch': epoch,
                          'lr': self.params['lr']}

            if epoch in self.params['save_on_epochs']:
                logger.info(f'Saving model on epoch {epoch}')
                self.save_checkpoint(
                    saved_dict, False, filename=f'{model_name}.epoch_{epoch}.run_{run_idx}')

    def save_model(self, model=None, epoch=0, val_loss=0):
        if model is None:
            model = self.target_model
        if self.params['save_model']:

            model_name = '{0}/model_last.pt.tar'.format(
                self.params['folder_path'])
            saved_dict = {'state_dict': model.state_dict(), 'epoch': epoch,
                          'lr': self.params['lr']}
            self.save_checkpoint(saved_dict, False, model_name)

            if epoch % 1 == 0:  # save at every epoch
                logger.info(f'Saving model on epoch {epoch}')
                self.save_checkpoint(
                    saved_dict, False, filename=f'{model_name}.epoch_{epoch}')
            if val_loss < self.best_loss:
                self.save_checkpoint(saved_dict, False, f'{model_name}.best')
                self.best_loss = val_loss

    @staticmethod
    def weighted_average_oracle(points, weights):
        """Computes weighted average of atoms with specified weights

        Args:
            points: list, whose weighted average we wish to calculate
                Each element is a list_of_np.ndarray
            weights: list of weights of the same length as atoms
        """
        tot_weights = torch.sum(weights)

        weighted_updates = dict()

        for name, data in points[0].items():
            weighted_updates[name] = torch.zeros_like(data)
        for w, p in zip(weights, points):  # 对每一个agent
            for name, data in weighted_updates.items():
                temp = (w / tot_weights).float().to(config.device)
                temp = temp * (p[name].float())
                # temp = w / tot_weights * p[name]
                if temp.dtype != data.dtype:
                    temp = temp.type_as(data)
                data.add_(temp)

        return weighted_updates

    @staticmethod
    def geometric_median_objective(median, points, alphas):
        """Compute geometric median objective."""
        temp_sum = 0
        for alpha, p in zip(alphas, points):
            temp_sum += alpha * Helper.l2dist(median, p)
        return temp_sum

        # return sum([alpha * Helper.l2dist(median, p) for alpha, p in zip(alphas, points)])

    def geometric_median_update(self, target_model, updates, maxiter=4, eps=1e-5, verbose=False, ftol=1e-6, max_update_norm=None):
        """Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
               """
        points = []
        alphas = []
        names = []
        for name, data in updates.items():
            points.append(data[1])  # update
            alphas.append(data[0])  # num_samples
            names.append(name)

        adver_ratio = 0
        for i in range(0, len(names)):
            _name = names[i]
            if _name in self.params['adversary_list']:
                adver_ratio += alphas[i]
        adver_ratio = adver_ratio / sum(alphas)
        poison_fraction = adver_ratio * \
            self.params['poisoning_per_batch'] / self.params['batch_size']
        logger.info(
            f'[rfa agg] training data poison_ratio: {adver_ratio}  data num: {alphas}')
        logger.info(
            f'[rfa agg] considering poison per batch poison_fraction: {poison_fraction}')

        alphas = np.asarray(alphas, dtype=np.float64) / sum(alphas)
        alphas = torch.from_numpy(alphas).float()

        # alphas.float().to(config.device)
        median = Helper.weighted_average_oracle(points, alphas)
        num_oracle_calls = 1

        # logging
        obj_val = Helper.geometric_median_objective(median, points, alphas)
        logs = []
        log_entry = [0, obj_val, 0, 0]
        logs.append(log_entry)
        if verbose:
            logger.info('Starting Weiszfeld algorithm')
            logger.info(log_entry)
        logger.info(f'[rfa agg] init. name: {names}, weight: {alphas}')
        # start

        weights = torch.tensor([alpha / max(eps, Helper.l2dist(median, p)) for alpha, p in zip(alphas, points)],
                               dtype=alphas.dtype)
        weights = weights / weights.sum()
        wv = copy.deepcopy(weights)
        for i in range(maxiter):
            prev_median, prev_obj_val = median, obj_val
            weights = torch.tensor([alpha / max(eps, Helper.l2dist(median, p)) for alpha, p in zip(alphas, points)],
                                   dtype=alphas.dtype)
            weights = weights / weights.sum()
            median = Helper.weighted_average_oracle(points, weights)
            num_oracle_calls += 1
            obj_val = Helper.geometric_median_objective(median, points, alphas)
            log_entry = [i + 1, obj_val,
                         (prev_obj_val - obj_val) / obj_val,
                         Helper.l2dist(median, prev_median)]
            logs.append(log_entry)
            if verbose:
                logger.info(log_entry)
            if abs(prev_obj_val - obj_val) < ftol * obj_val:
                break
            logger.info(
                f'[rfa agg] iter:  {i}, prev_obj_val: {prev_obj_val}, obj_val: {obj_val}, abs dis: { abs(prev_obj_val - obj_val)}')
            logger.info(f'[rfa agg] iter:  {i}, weight: {weights}')
            wv = copy.deepcopy(weights)
        alphas = [Helper.l2dist(median, p) for p in points]

        update_norm = 0
        for name, data in median.items():
            update_norm += torch.sum(torch.pow(data, 2))
        update_norm = math.sqrt(update_norm)

        if max_update_norm is None or update_norm < max_update_norm:
            for name, data in target_model.state_dict().items():
                update_per_layer = median[name]   # *(self.params["eta"]=1)
                # if self.params['diff_privacy']:
                #     update_per_layer.add_(self.dp_noise(data, self.params['sigma']))
                data.add_(update_per_layer)
            is_updated = True
        else:
            logger.info(
                '\t\t\tUpdate norm = {} is too large. Update rejected'.format(update_norm))
            is_updated = False

        # utils.csv_record.add_weight_result(names, wv.cpu().numpy().tolist(), alphas)

        return num_oracle_calls, is_updated, names, wv.cpu().numpy().tolist(), alphas
