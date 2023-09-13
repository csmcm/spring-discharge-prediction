from train import *
import argparse
import torch


def wst_norm(wst):
    shape = wst.shape
    wst_min = wst.min(axis=(1, 2)).repeat(shape[1], axis=-1).repeat(shape[2], axis=-1).reshape(shape)
    wst_max = wst.max(axis=(1, 2)).repeat(shape[1], axis=-1).repeat(shape[2], axis=-1).reshape(shape)
    wst_norm = (wst-wst_min)/(wst_max-wst_min)
    return wst_norm


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=cfg['gpu'])
    parser.add_argument('--source_file', type=str, default=cfg['source_file'])
    parser.add_argument('--best_model_dir', type=str, default=cfg['best_model_dir'])
    args = parser.parse_args()

    device = torch.device('cpu')

    # experiment_list = [[0, 12], [1, 12], [2, 12], [3, 0], [3, 4], [3, 8], [3, 12], [3, 16], [3, 20], [4, 12], [5, 12]]
    experiment_list = [[3, 12]]

    # 评价模型
    # model_path_list = get_model_path(args.best_model_dir)
    path_list = get_model_path('./experiment model')
    model_path_list = []
    for e in experiment_list:
        for path in path_list:
            s = 'long%02d_short%02d'%(e[0],e[1])
            file_name = os.path.basename(path)
            if s == file_name[0:14]:
                model_path_list.append(path)


    with torch.no_grad():
        for model_path,e in zip(model_path_list,experiment_list):
            print('t_long=%02d, t_short=%02d:' % (e[0], e[1]))
            cfg['num_past_year'] = e[0]
            cfg['num_past_month'] = e[1]
            data_generator = DataGenerator(args.source_file, 1)
            num_past_month = cfg['num_past_month']
            num_region = cfg['num_region']

            print(str(model_path), ':')
            title_list = [' ', 'NSE', 'RMSE', 'MAE', 'MAPE', 'R2']
            training_result = ['training']
            testing_result = ['testing']
            result = [training_result, testing_result]
            model_evl = get_model('testing', device, model_path)
            period_list = ['training', 'testing']
            weight_spatial_dic = {'training': [], 'testing': []}
            weight_temporal_dic = {'training': [], 'testing': []}
            for period, result in zip(period_list, result):
                data_generator.batch_size = 1
                data_generator.mode = period
                true_list = []
                pred_list = []
                while not data_generator.is_epoch_end():
                    data = data_generator()
                    # 预测
                    result_dic = model_evl(data)
                    w_t = result_dic['w_t'].squeeze(dim=0)  # [7, 12]
                    w_t = w_t.permute(1, 0).contiguous()  # [12, 7]
                    weight_spatial_dic[period].append(result_dic['w_s'].squeeze(dim=0))
                    weight_temporal_dic[period].append(w_t.cpu().numpy())
                    true_list.append(result_dic['data_lab'])
                    pred_list.append(result_dic['out'])
                true = torch.cat(true_list, dim=0).view(-1)
                pred = torch.cat(pred_list, dim=0).view(-1)
                f_save = open('./result/long%02d_short%02d_%s_true.pkl' % (cfg['num_past_year'],cfg['num_past_month'],period), 'wb')
                pickle.dump(true.unsqueeze(dim=-1).cpu().numpy(), f_save)
                f_save.close()
                f_save = open('./result/long%02d_short%02d_%s_pred.pkl' % (cfg['num_past_year'],cfg['num_past_month'],period), 'wb')
                pickle.dump(pred.unsqueeze(dim=-1).cpu().numpy(), f_save)
                f_save.close()
                weight_spatial_dic[period] = torch.cat(weight_spatial_dic[period], dim=0).cpu().numpy()
                for key, criterion in criterion_dic.items():
                    result.append(criterion(true, pred))
            f_save = open('./result/long%02d_short%02d_w_s.pkl' % (cfg['num_past_year'], cfg['num_past_month']), 'wb')
            pickle.dump(weight_spatial_dic, f_save)
            f_save.close()
            f_save = open('./result/long%02d_short%02d_w_t.pkl' % (cfg['num_past_year'], cfg['num_past_month']), 'wb')
            pickle.dump(weight_temporal_dic, f_save)  # 不同阶段（字典）——对过去12个月的注意力（列表）
            f_save.close()

            print('-' * 60)
            print('%10s\t%4s\t%4s\t%4s\t%4s\t%4s' % tuple(title_list))
            print('%10s\t%04.2f\t%04.2f\t%04.2f\t%04.2f\t%04.2f' % tuple(training_result))
            print('%10s\t%04.2f\t%04.2f\t%04.2f\t%04.2f\t%04.2f' % tuple(testing_result))
            print('-' * 60)
