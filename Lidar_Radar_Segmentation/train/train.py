from mmengine.config import Config
from mmdet3d.apis import train_model
from mmdet3d.registry import init_model, build_dataset
from mmengine.runner import Runner

def main():
    # Carica la configurazione base (puoi modificarla da script)
    cfg = Config.fromfile("projects/pointpillars/config.py")

    # Modifiche alla configurazione
    cfg.work_dir = "./work_dirs/pointpillars"
    cfg.load_from = None  # Oppure path ai pesi preaddestrati se li hai

    # Se stai usando TruckScenes o un dataset personalizzato:
    # cambia qui la config (es. dataset_type, data_root, ecc.)

    # Costruisci il dataset
    datasets = [build_dataset(cfg.train_dataloader.dataset)]

    # Inizializza il modello
    model = init_model(cfg.model, cfg, device="cuda")

    # Allena
    train_model(model, datasets, cfg)

if __name__ == '__main__':
    main()
