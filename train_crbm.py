import os
import time
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import argparse
from torch.utils.data import Dataset, DataLoader


from cRBM_class import cRBM
from h5_class import save_checkpoint_h5


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True 

## Dataset para ventana deslizante
class cRBMDataset(Dataset):
    def __init__(self, data, n_past):
        """
        data: Tensor secuencial de shape (L, n_vis)
        n_past: Número de frames pasados para usar como contexto 'u'
        """
        self.data = data
        self.n_past = n_past
        self.n_vis = data.shape[1]

    def __len__(self):
        # Si tenemos L frames, podemos extraer (L - n_past) ventanas completas
        return len(self.data) - self.n_past

    def __getitem__(self, idx):
        # u: extraemos los frames pasados y los aplanamos en un vector 1D
        u = self.data[idx : idx + self.n_past].view(-1)
        # v_t: el frame objetivo actual
        v_t = self.data[idx + self.n_past]
        return u, v_t





def main():
    parser = argparse.ArgumentParser(description="Training of cRBM")
    parser.add_argument("--data", type=str, default="data/data_binary_DT_0.02_sessionid_1108335514_mouseid_571520_Familiar_Truncado_120.h5", 
                        help=".h5 dataset path (NO STACKED)")
    parser.add_argument("--batch", type=int, default=512, help="Tamaño del batch")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--updates", type=int, default=1000000, help="Number of updates")
    parser.add_argument("--checkpoints", type=int, default=50, help="Number of log checkpoints")
    parser.add_argument("--hidden", type=int, default=300, help="number of hidden nodes")
    parser.add_argument("--past", type=int, default=10, help="Number of past frames for context (n_past)")
    parser.add_argument("--steps", type=int, default=1, help="Number of CD steps (typically 1 for cRBM)")
    parser.add_argument("--wdecay", type=float, default=0.0001, help="Weight decay")
    
    args = parser.parse_args()

    # Comprobación base de datos
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
    N_PAST = args.past
    N_STEP = args.steps
    WEIGHT_DECAY = args.wdecay
    N_RECORTE = 120

    # Creación de carpetas
    os.makedirs("models/H5", exist_ok=True)
    os.makedirs("models/PTH", exist_ok=True)
    os.makedirs("models/energy_curve", exist_ok=True)

    H5_HISTORY_FILE = f"models/H5/history_crbm_{N_STEP}_{N_UPDATES}_{BATCH_SIZE}_{N_PAST}_{N_HIDDEN}_{LEARNING_RATE}.h5"      
    LOG_MILESTONES = sorted(set(np.logspace(0, np.log10(N_UPDATES), N_CHCK).astype(int))) 
    
    # Cargamos los datos originales (sin pre-stacking)
    print(f"Cargando dataset desde {DATA_FILE}...")
    with h5py.File(DATA_FILE, 'r') as f:
        # Suponemos que los datos en el h5 son matrices 2D (Samples, Píxeles)
        train_data_cpu = torch.tensor(f['train'][:], dtype=torch.float32)
        test_data_cpu = torch.tensor(f['test'][:], dtype=torch.float32)
        N_VIS_TOTAL = train_data_cpu.shape[1]

    # a la GPU
    print("Transfiriendo dataset a la VRAM...")
    train_data_gpu = train_data_cpu.to(device)
    test_data_gpu = test_data_cpu.to(device)
        
    print(f"Datos listos.")
    print(f"   -> Train Frames: {len(train_data_gpu)}")
    print(f"   -> Test Frames:  {len(test_data_gpu)}")
    print(f"   -> Frame dimension (n_vis): {N_VIS_TOTAL}")
    print(f"   -> Context window (n_past): {N_PAST} frames")

    # optiimizacion
    # Nota: Usamos shuffle=True para barajar las VENTANAS extraídas, 
    train_dataset = cRBMDataset(train_data_gpu, N_PAST)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # Creamos un batch de test fijo para la validación 
    test_dataset = cRBMDataset(test_data_gpu, N_PAST)
    test_loader_fixed = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    test_u_batch, test_v_batch = next(iter(test_loader_fixed))

    # Inicializamos el modelo 
    rbm = cRBM(n_vis=N_VIS_TOTAL, n_hid=N_HIDDEN, n_past=N_PAST,
               learning_rate=LEARNING_RATE, momentum=0.5, 
               weight_decay=WEIGHT_DECAY, device=device.type)
    
    # Aquí podríamos añadir una inicialización estadística para los sesgos base si quisieramos
    
    with h5py.File(H5_HISTORY_FILE, 'w') as f:
        f.attrs['hyperparameters'] = str({'lr': LEARNING_RATE, 'n_hid': N_HIDDEN, 'n_past': N_PAST})



    # Training loop   
    train_energy_history = []
    test_energy_history = []
    recent_energies = []
    start_train = time.time()
    
    update_counter = 0

    try:
        with tqdm(total=N_UPDATES, desc="Training cRBM", unit="upd") as pbar:
            while update_counter < N_UPDATES:
                # Iteramos sobre el DataLoader
                for u_batch, v_batch in train_loader:
                    if update_counter >= N_UPDATES:
                        break
                    
                    # Llamamos a train_step_cd (pasándole v_t y u)
                    energy = rbm.train_step_cd(v_data=v_batch, u_data=u_batch, k=N_STEP)
                    
                    recent_energies.append(energy)
                
                    update_counter += 1 
                    pbar.update(1)
                    
                    # Checkpoints
                    if update_counter in LOG_MILESTONES:
                        with torch.no_grad():
                            curr_test_E = torch.mean(rbm.free_energy(test_v_batch, test_u_batch)).item()
                        save_checkpoint_h5(H5_HISTORY_FILE, rbm, update_counter, energy, curr_test_E)

                    # Logging y Validación
                    if update_counter % LOG_INTERVAL == 0:
                        train_avg = np.mean(recent_energies)
                        train_energy_history.append(train_avg)
                        recent_energies = []

                        with torch.no_grad():
                            test_E = torch.mean(rbm.free_energy(test_v_batch, test_u_batch)).item()
                            test_energy_history.append(test_E) 

                        pbar.set_postfix(
                            Train_E=f"{train_avg:.1f}", 
                            Test_E=f"{test_E:.1f}"
                        )
                    
    except KeyboardInterrupt:
        print("\n Entrenamiento interrumpido por el usuario.")

    print(f"Tiempo total: {time.time()-start_train:.2f}s")
    
   
   
   
   
   
    model_path = f"models/PTH/crbm_{N_STEP}_{N_UPDATES}_{BATCH_SIZE}_{N_PAST}_{N_HIDDEN}_{LEARNING_RATE}.pth"
    torch.save({
        'W': rbm.W, 
        'A': rbm.A,
        'B': rbm.B,
        'vbias': rbm.vbias, 
        'hbias': rbm.hbias,
        'hyperparameters': {'n_vis': N_VIS_TOTAL, 'n_hid': N_HIDDEN, 'n_past': N_PAST}
    }, model_path)
    print(f"Modelo final guardado en: {model_path}")

    
    plt.figure(figsize=(10, 5))
    plt.plot(train_energy_history, label='Train Energy', color='blue', alpha=0.6)
    plt.plot(test_energy_history, label='Test Energy', color='red', linewidth=2)
    plt.xlabel(f'Updates (x{LOG_INTERVAL})')
    plt.ylabel('Free Energy (Menos es mejor)')
    plt.title(f'Curva de Aprendizaje - cRBM')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = f'models/energy_curve/energy_curve_crbm_{N_STEP}_{N_UPDATES}_{BATCH_SIZE}_{N_PAST}_{N_HIDDEN}_{LEARNING_RATE}_{N_RECORTE}.png'
    plt.savefig(plot_path)
    print(f"Gráfica de convergencia guardada en: {plot_path}")

if __name__ == "__main__":
    main()
