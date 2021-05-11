import pickle
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
use_gpu = torch.cuda.is_available()

def get_base_class_statistics(file):
    base_means = []
    base_cov = []
    data = pickle.load(file)
    for key in data.keys():
        feature = torch.tensor(data[key])
        base_means.append(torch.mean(feature, axis = 0))
        np_feature = np.array(data[key])
        np_cov = np.cov(np_feature.T)
        cov = torch.from_numpy(np_cov)
        base_cov.append(cov)
    
    torch_means = torch.zeros(len(base_means), len(base_means[0]))
    torch_cov = torch.zeros(len(base_cov), len(base_cov[0]), len(base_cov[0][0]))
    for i in range(len(base_means)):
        torch_means[i] = torch.tensor(base_means[i])
        for j in range(len(base_cov[0])):
            torch_cov[i][j] = torch.tensor(base_cov[i][j])
    
    return torch_means, torch_cov

def transform_tukey(data, lam):
    transformed_data = torch.zeros(data.shape)
    for i in range(len(data)):
        feature = data[i]
        if lam != 0:
            transformed_data[i] = feature ** lam
        else:
            transformed_data[i] = torch.log(feature)
    return transformed_data

def calibrate(base_means, base_cov, feature, k, alpha):
    distances = torch.zeros(len(base_means))
    for i in range(len(base_means)):
        distances[i] = torch.cdist(feature.unsqueeze(0), base_means[i].unsqueeze(0))
    _, indices = torch.topk(distances, k)
    #selected_means = torch.tensor([base_means[i] for i in indices][0])
    selected_means = torch.index_select(base_means, 0, indices)
    selected_cov = torch.index_select(base_cov, 0, indices)
    #print(selected_cov[0], '\n')
    #print(selected_cov[1], '\n')
    #print(selected_cov[0][0][0] + selected_cov[1][0][0])
    calibrated_mean = (torch.sum(selected_means, axis=0) + feature) / (k + 1)
    calibrated_cov = torch.sum(selected_cov, axis=0) / k + alpha
    return calibrated_mean, calibrated_cov
    
if __name__ == '__main__':
    # ---- data loading
    dataset = 'miniImagenet'
    n_shot = 1
    n_ways = 5
    n_queries = 15
    n_runs = 5
    n_train = n_ways * n_shot
    n_test = n_ways * n_queries
    n_total = n_train + n_test
    lam = 0.5
    alpha = 0.21
    n_generation = int(750 / n_shot)
    
    import FSLTask
    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}
    FSLTask.loadDataSet(dataset)
    FSLTask.setRandomStates(cfg)
    ndatas = FSLTask.GenerateRunSet(end=n_runs, cfg=cfg)
    print(ndatas.shape)
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_total, -1)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 5).clone().view(n_runs,
                                                                                                        n_total)
    print(labels.shape)
    # ---- Base class statistics
    base_means = None
    base_cov = None
    base_features_path = "./checkpoints/%s/WideResNet28_10_S2M2_R/last/base_features.plk"%dataset
    with open(base_features_path, 'rb') as input_file:
        base_means, base_cov = get_base_class_statistics(input_file)
        
    from torch.distributions.multivariate_normal import MultivariateNormal
    
    acc = []
    print (ndatas.shape)
    for task, labels in tqdm(zip(ndatas, labels)):
        tukey_data = transform_tukey(task, lam)
        # separate train data and labels from tests
        support_data = tukey_data[:n_train]
        support_labels = labels[:n_train]
        test_data = tukey_data[n_train:]
        test_labels = labels[n_train:]
        
        train_data = support_data.clone()
        train_labels = support_labels.clone()
        for feature, label in zip(support_data, support_labels):
            calibrated_mean, calibrated_cov = calibrate(base_means, base_cov, feature, 2, alpha)
            calibrated_distribution = MultivariateNormal(calibrated_mean, calibrated_cov)
            sampled_data = torch.zeros(n_generation, len(calibrated_mean))
            sampled_labels = torch.zeros(n_generation)
            for i in range(n_generation):
                sampled_data[i] = calibrated_distribution.sample()
                sampled_labels[i] = label
            train_data = torch.cat((train_data, sampled_data), dim=0)
            train_labels = torch.cat((train_labels, sampled_labels), dim=0)
        
        classifier = LogisticRegression(max_iter=1000).fit(X=train_data, y=train_labels)
        predicts = classifier.predict(test_data)
        count = 0
        for predict, true in zip(predicts, test_labels):
            if predict == true:
                count += 1
        print(count/len(predicts))
        #print(support_data.shape, sampled_data.shape)
        #train_data = torch.cat((support_data, sampled_data), dim=0)
        #train_labels = torch.cat((support_labels, sampled_labels), dim=0)
        #print (train_data.shape, train_labels.shape)