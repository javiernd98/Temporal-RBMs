import os
import time
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import argparse
from torch.utils.data import TensorDataset, DataLoader

from RBM_class import RBM_PCD_Energy
from h5_class import save_checkpoint_h5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True 

def main():
    parser = argparse.ArgumentParser(description="Training of RBM")
    parser.add_argument("--data", type=str, default="datos stack/Neuro_Allen_Stacked_12_120.h5", 
                        help=".h5 dataset path")
    parser.add_argument("--batch", type=int, default=1024, help="Tamaño del batch")
    parser.add_argument("--lr", type=float, default=0.0008, help="Learning rate")
    parser.add_argument("--updates", type=int, default=1000000, help="Number of updates")
    parser.add_argument("--checkpoints", type=int, default=50, help="Number of log checkpoints")
    parser.add_argument("--hidden", type=int, default=500, help="number of hidden nodes")
    parser.add_argument("--steps", type=int, default=10, help="Number of MCMC steps per update")
    parser.add_argument("--wdecay", type=float, default=0.0001, help="Weight decay")
    
    args = parser.parse_args()

    DATA_FILE = args.data  
    
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"'{DATA_FILE}' NOT FOUND.")

    # Hiperparámetros de entrenamiento
    BATCH_SIZE = args.batch
    LEARNING_RATE = args.lr
    N_UPDATES = args.updates
    LOG_INTERVAL = 1000 
    N_CHCK = args.checkpoints 
    N_HIDDEN = args.hidden
    N_STEP = args.steps
    WEIGHT_DECAY = args.wdecay

    # Creación de carpetas si no existen
    os.makedirs("models/H5", exist_ok=True)
    os.makedirs("models/PTH", exist_ok=True)
    os.makedirs("models/energy_curve", exist_ok=True)

    H5_HISTORY_FILE = f"models/H5/history_rbm_{N_UPDATES}_{BATCH_SIZE}_{N_STEP}_{N_HIDDEN}_{LEARNING_RATE}.h5"      
    LOG_MILESTONES = sorted(set(np.logspace(0, np.log10(N_UPDATES), N_CHCK).astype(int))) 
    
    # Cargamos los datos
    print(f"Cargando dataset desde {DATA_FILE}...")
    with h5py.File(DATA_FILE, 'r') as f:
        train_data_cpu = torch.tensor(f['train'][:], dtype=torch.float32)
        test_data_cpu = torch.tensor(f['test'][:], dtype=torch.float32)
        
        if 'hyperparameters' in f and 'n_stack' in f['hyperparameters']:
            N_STACK = int(f['hyperparameters']['n_stack'][()])
        else:
            raise KeyError(f"ERROR: 'n_stack' no encontrado en {DATA_FILE}. "
                           "El entrenamiento no puede continuar sin este parámetro.")
            
        N_VIS_TOTAL = train_data_cpu.shape[1]

    # a la GPU
    print("Transfiriendo dataset a la VRAM...")
    train_data_gpu = train_data_cpu.to(device)
    test_data_gpu = test_data_cpu.to(device)
        
    print(f"Datos listos.")
    print(f"   -> Train Samples: {len(train_data_gpu)}")
    print(f"   -> Test Samples:  {len(test_data_gpu)}")
    print(f"   -> Visible layer dimension: {N_VIS_TOTAL}")
    print(f"   -> Stacking of {N_STACK} frames")

    # Optimizacion
    train_dataset = TensorDataset(train_data_gpu)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    
    rbm = RBM_PCD_Energy(n_vis_total=N_VIS_TOTAL, n_hid=N_HIDDEN, n_stack=N_STACK,
                         learning_rate=LEARNING_RATE, momentum=0.5, 
                         weight_decay=WEIGHT_DECAY, device=device.type)
    
    rbm.init_biases_statistics(train_data_gpu) 
    
    with h5py.File(H5_HISTORY_FILE, 'w') as f:
        f.attrs['hyperparameters'] = str({'lr': LEARNING_RATE, 'n_hid': N_HIDDEN, 'stack': N_STACK})

    # Training loop 
    train_energy_history = []
    test_energy_history = []
    recent_energies = []
    start_train = time.time()
    
    update_counter = 0

    try:
        with tqdm(total=N_UPDATES, desc="Training RBM", unit="upd") as pbar:
            while update_counter < N_UPDATES:
                # Iteramos sobre el DataLoader directamente
                for (batch,) in train_loader:
                    if update_counter >= N_UPDATES:
                        break
                    
                    energy = rbm.train_step_pcd(batch, k=N_STEP)
                    
                    # .item() para evitar memory leaks en la VRAM
                    recent_energies.append(energy.item() if torch.is_tensor(energy) else energy)
                
                    update_counter += 1 
                    pbar.update(1)
                    
                    # Checkpoints
                    if update_counter in LOG_MILESTONES:
                        with torch.no_grad():
                            t_batch = test_data_gpu[:min(1000, len(test_data_gpu))]
                            curr_test_E = torch.mean(rbm.free_energy(t_batch)).item()
                        save_checkpoint_h5(H5_HISTORY_FILE, rbm, update_counter, energy, curr_test_E)

                    # Logging y Validación
                    if update_counter % LOG_INTERVAL == 0:
                        train_avg = np.mean(recent_energies)
                        train_energy_history.append(train_avg)
                        recent_energies = []

                        with torch.no_grad():
                            t_batch = test_data_gpu[:BATCH_SIZE]
                            test_E = torch.mean(rbm.free_energy(t_batch)).item()
                            test_energy_history.append(test_E) 

                        pbar.set_postfix(
                            Train_E=f"{train_avg:.1f}", 
                            Test_E=f"{test_E:.1f}"
                        )
                    
    except KeyboardInterrupt:
        print("\n Entrenamiento interrumpido por el usuario.")

    print(f"Tiempo total: {time.time()-start_train:.2f}s")
    
    # --- Guardado final --- 
    model_path = f"models/PTH/rbm_{N_UPDATES}_{BATCH_SIZE}_{N_STEP}_{N_HIDDEN}_{LEARNING_RATE}.pth"
    torch.save({
        'W': rbm.W, 
        'vbias': rbm.vbias, 
        'hbias': rbm.hbias,
        'hyperparameters': {'n_vis': N_VIS_TOTAL, 'n_hid': N_HIDDEN, 'n_stack': N_STACK}
    }, model_path)
    print(f"Modelo final guardado en: {model_path}")

    # --- GRÁFICA DE ENERGÍA ---
    plt.figure(figsize=(10, 5))
    plt.plot(train_energy_history, label='Train Energy', color='blue', alpha=0.6)
    plt.plot(test_energy_history, label='Test Energy', color='red', linewidth=2)
    plt.xlabel(f'Updates (x{LOG_INTERVAL})')
    plt.ylabel('Free Energy (Menos es mejor)')
    plt.title(f'Curva de Aprendizaje')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = f'models/energy_curve/energy_curve_{N_UPDATES}_{BATCH_SIZE}_{N_STEP}_{N_HIDDEN}_{LEARNING_RATE}.png'
    plt.savefig(plot_path)
    print(f"Gráfica de convergencia guardada en: {plot_path}")

if __name__ == "__main__":
    main()

    
