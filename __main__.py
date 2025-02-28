from src.data.dataset import SpeakerDataset, UbmGmmDataset
from src.utils.config_manager import ConfigManager
from src.models.svm import SVMClassifier
from src.models.random_forest import RandomForest
from src.models.gmm_ubm import GMM_UBM
conf = ConfigManager('configs/conf.json')

def test_dataset(data_path, speakers):
    data = SpeakerDataset(data_path, speakers)
    print(data.df.columns)
    print(data.df.size)
    print(f"features: {type(data.features)}, {data.features.shape}")
    print(f"labels: {data.labels}, {data.decode_label(data.labels)[0]}")

    return data

def test_svm(dataset, kernel='linear'):
    save_path = conf.get('svm.save_path')
    filename = 'svm.pkl'

    svm = SVMClassifier(dataset=dataset, kernel=kernel, test_size=0.1, val_size=0.1)
    svm.train()
    svm.evaluate()
    svm.save(save_path, filename)
    # svm.load(save_path, filename)
    # svm.predict()

def test_rf(dataset, n_estimators, test_size: int = 0.1, val_size : int = 0.1):
    save_path = conf.get('random_forest.save_path')
    filename = 'random_forest.pkl'

    rf = RandomForest(dataset, n_estimators=n_estimators, test_size=test_size, val_size=val_size)
    rf.train()
    rf.evaluate()
    rf.save(save_path,filename)
    # rf.load(save_path, filename)
    # rf.predict()

def test_ubm_gmm(dataset, n_components, max_iter, covariance_type, random_state, gmm_size: int = 0.3, test_size: int = 0.2):
    save_path = conf.get('ubm_gmm.save_path')
    filename = 'ubm_gmm.pkl'
    ubm_gmm = GMM_UBM(
        dataset=dataset,
        gmm_size=gmm_size,
        test_size=test_size,
        n_components=n_components,
        max_iter=max_iter,
        covariance_type=covariance_type,
        random_state=random_state
    )

    ubm_gmm.build()
    ubm_gmm.train()
    ubm_gmm.evaluate()
    ubm_gmm.save(save_path, filename)



if __name__ == '__main__':
    database = conf.get('database.path')
    speakers = conf.get('database.speakers')
    dataset = test_dataset(database, speakers)

    test_svm(dataset, kernel='linear')
    test_rf(dataset, n_estimators=100)

    n_components = conf.get('ubm_gmm.n_components')
    max_iter = conf.get('ubm_gmm.max_iter')
    covariance_type = conf.get('ubm_gmm.covariance_type')
    random_state = conf.get('ubm_gmm.random_state')

    dataset_ubm = UbmGmmDataset(data_path=database, speakers=speakers)
    test_ubm_gmm(dataset_ubm, n_components=n_components, max_iter=max_iter, covariance_type=covariance_type, random_state=random_state)




