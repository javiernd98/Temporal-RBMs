import numpy as np
import torch
import os
import glob
from tqdm import tqdm

class cRBM:
    def __init__(self, n_vis: int, n_hid: int, n_past: int, learning_rate=0.001, momentum=0.5, weight_decay=0.0001, device="cpu"):
        self.device = torch.device(device)
        self.n_vis = n_vis                 # Tamaño de un frame real
        self.n_hid = n_hid                 # Nodos ocultos
        self.n_past = n_past               # tamaño de la ventana temporal para el entrenamiento
        
        # Aquí ajustamos las dimensiones de u
        self.history_size = self.n_vis * self.n_past 
        
        self.lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        
        # W Conecta v_t (n_vis) con h_t (n_hid), esta es la matriz que encontramos en cualquier RBM estándar
        self.W = torch.randn(self.n_vis, self.n_hid, device=self.device) * 0.01
        
        # Conecta u (history_size) con v_t (n_vis), es decir, los vectores del tiempo presente con el número de tiempos pasados que hayamos prefijado
        self.A = torch.randn(self.history_size, self.n_vis, device=self.device) * 0.01
        
        # Conecta u (history_size) con h_t (n_hid)
        self.B = torch.randn(self.history_size, self.n_hid, device=self.device) * 0.01

        
        self.vbias = torch.zeros(self.n_vis, device=self.device)
        self.hbias = torch.zeros(self.n_hid, device=self.device)

        
        self.vW = torch.zeros_like(self.W)
        self.vA = torch.zeros_like(self.A)
        self.vB = torch.zeros_like(self.B)
        self.vvbias = torch.zeros_like(self.vbias)
        self.vhbias = torch.zeros_like(self.hbias)

    def _sigmoid(self, x):
        return torch.sigmoid(x)

    def get_dynamic_biases(self, u):
        # Calculamos los sesgos condicionados al contexto 'u'
        dynamic_vbias = self.vbias + torch.matmul(u, self.A)
        dynamic_hbias = self.hbias + torch.matmul(u, self.B)
        
        return dynamic_vbias, dynamic_hbias

    def sample_h(self, v_t, dynamic_hbias):
        # Muestreo oculto condicionado
        mh = self._sigmoid(dynamic_hbias + torch.matmul(v_t, self.W))
        return torch.bernoulli(mh), mh

    def sample_v(self, h_t, dynamic_vbias):
        # Muestreo visible condicionado
        mv = self._sigmoid(dynamic_vbias + torch.matmul(h_t, self.W.t()))
        return torch.bernoulli(mv), mv

    def free_energy(self, v_t, u):
        dynamic_vbias, dynamic_hbias = self.get_dynamic_biases(u)
        
        vbias_term = torch.sum(v_t * dynamic_vbias, dim=1)
        wx_b = torch.matmul(v_t, self.W) + dynamic_hbias
        hidden_term = torch.sum(torch.nn.functional.softplus(wx_b), dim=1)
        
        return -hidden_term - vbias_term

    def train_step_cd(self, v_data, u_data, k=1):
        batch_size = v_data.shape[0]

        # Aquí precalculamos los sesgos dinámicos
        dynamic_vbias, dynamic_hbias = self.get_dynamic_biases(u_data)


        # Calculamos la probabilidad de activación oculta con los datos reales (particula positiva)
        _, mh_data = self.sample_h(v_data, dynamic_hbias)

        # Particula negativa (iniciamos la cadena con los datos reales
        v_chain = v_data.clone()

        for _ in range(k):
            h_chain, _ = self.sample_h(v_chain, dynamic_hbias) # h dado v y u
            v_chain, _ = self.sample_v(h_chain, dynamic_vbias) # v dado h y u

        _, mh_chain = self.sample_h(v_chain, dynamic_hbias)

       
       
        pos_grad_W = torch.matmul(v_data.t(), mh_data)
        neg_grad_W = torch.matmul(v_chain.t(), mh_chain)
        grad_W = (pos_grad_W - neg_grad_W) / batch_size - (self.weight_decay * self.W)

        grad_vbias = (v_data.sum(0) - v_chain.sum(0)) / batch_size
        grad_hbias = (mh_data.sum(0) - mh_chain.sum(0)) / batch_size

        grad_A = torch.matmul(u_data.t(), (v_data - v_chain)) / batch_size
        grad_B = torch.matmul(u_data.t(), (mh_data - mh_chain)) / batch_size

       
        self.vW = self.momentum * self.vW + self.lr * grad_W
        self.vvbias = self.momentum * self.vvbias + self.lr * grad_vbias
        self.vhbias = self.momentum * self.vhbias + self.lr * grad_hbias
        
        self.vA = self.momentum * self.vA + self.lr * grad_A
        self.vB = self.momentum * self.vB + self.lr * grad_B

        self.W += self.vW
        self.vbias += self.vvbias
        self.hbias += self.vhbias
        self.A += self.vA
        self.B += self.vB

        return torch.mean(self.free_energy(v_data, u_data)).item()

    def generate_sequence(self, seed_frames, n_steps_future, gibbs_steps=10):
        """
        Genera secuencias futuras de forma autorregresiva.
        
        Parámetros:
        - seed_frames: Tensor de shape (batch_size, n_past, n_vis). 
                       Es el contexto inicial ("cebo") para empezar a generar.
        - n_steps_future: Número de frames que queremos predecir hacia el futuro.
        - gibbs_steps: Cuántos pasos de MCMC dar para asentar cada nuevo frame.
        
        Retorna:
        - Secuencia completa generada (semilla + futuro) de shape (batch_size, n_past + n_steps_future, n_vis)
        """
        
        
        generated_seq = seed_frames.clone().to(self.device)
        batch_size = generated_seq.shape[0]

        for step in range(n_steps_future):
            # Extraemos los últimos 'n_past' frames para formar el contexto 'u'
            # Extraemos shape: (batch_size, n_past, n_vis) y aplanamos a (batch_size, history_size)
            u = generated_seq[:, -self.n_past:, :].reshape(batch_size, -1)
            
            # Calcular los sesgos dinámicos condicionados a este 'u'
            dynamic_vbias, dynamic_hbias = self.get_dynamic_biases(u)
            
            # Inicializar el frame actual v_t (copiamos el último frame como punto de partida)
            # Esto ayuda a que el sampling converja más rápido que partiendo de ruido
            v_t = generated_seq[:, -1, :].clone()
            
            # Pasos de Gibbs (Fase de alucinación condicionada)
            for _ in range(gibbs_steps):
                h, _ = self.sample_h(v_t, dynamic_hbias)
                v_t, _ = self.sample_v(h, dynamic_vbias)
                
            # v_t ahora tiene shape (batch_size, n_vis). Lo expandimos a (batch_size, 1, n_vis)
            v_t_expanded = v_t.unsqueeze(1)
            
            # Añadimos el nuevo frame a la secuencia temporal
            generated_seq = torch.cat((generated_seq, v_t_expanded), dim=1)
            
        return generated_seq

def generate_conditional_sequences(rbm, dataset, num_sequences=50, steps_to_predict=100, gibbs_steps=10, init_type='test', batch_size=200):
    """
    Genera secuencias autorregresivas usando la cRBM.
    
    Parámetros:
    - rbm: El modelo cRBM ya entrenado.
    - dataset: Diccionario con los datos reales {'train': train_data, 'test': test_data}
    - num_sequences: Número de secuencias independientes a generar.
    - steps_to_predict: Cuántos frames hacia el futuro queremos soñar.
    - gibbs_steps: Pasos de MCMC para estabilizar cada frame predicho.
    - init_type: 'test', 'train' o 'random'.
    - batch_size: Cuántas secuencias generar en paralelo a la vez para no saturar la VRAM.
    
    Retorna:
    - Array Numpy con shape (num_sequences, n_past + steps_to_predict, n_vis)
    """
    rbm.W = rbm.W.to(rbm.device)
    rbm.A = rbm.A.to(rbm.device)
    rbm.B = rbm.B.to(rbm.device)
    rbm.vbias = rbm.vbias.to(rbm.device)
    rbm.hbias = rbm.hbias.to(rbm.device)
    
    n_past = rbm.n_past
    generated_sequences = []
    
    print(f" Iniciando generación de {num_sequences} secuencias (cRBM)...")
    print(f"   - Origen semilla: {init_type.upper()} ({n_past} frames de contexto)")
    print(f"   - Pasos a predecir: {steps_to_predict}")
    print(f"   - Pasos Gibbs por frame: {gibbs_steps}")
    print("-" * 50)

    # Preparar el proceso en lotes (batches) para aprovechar la GPU al máximo
    # sin quedarnos sin memoria de vídeo.
    num_batches = int(np.ceil(num_sequences / batch_size))
    pbar = tqdm(total=num_sequences, desc="Generando", unit="seq")
    
    with torch.no_grad():
        for b in range(num_batches):
            # Cuántas secuencias tocan en este batch
            current_batch_size = min(batch_size, num_sequences - b * batch_size)
            
            # 1. Recopilar las semillas para el batch actual
            seeds_list = []
            for _ in range(current_batch_size):
                if init_type == 'train':
                    # Elegimos un índice aleatorio dejando margen para extraer 'n_past' frames
                    idx = torch.randint(0, len(dataset['train']) - n_past, (1,)).item()
                    seed = dataset['train'][idx : idx + n_past].clone()
                elif init_type == 'test':
                    idx = torch.randint(0, len(dataset['test']) - n_past, (1,)).item()
                    seed = dataset['test'][idx : idx + n_past].clone()
                elif init_type == 'random':
                    # Ruido puro para ver si la máquina converge a dinámicas reales desde el caos
                    seed = torch.bernoulli(torch.rand((n_past, rbm.n_vis)))
                else:
                    raise ValueError("init_type debe ser 'train', 'test' o 'random'")
                
                seeds_list.append(seed)
            
            # Apilamos las semillas: Shape -> (current_batch_size, n_past, n_vis)
            seed_batch = torch.stack(seeds_list).to(rbm.device)
            
            # Generar el futuro de TODAS las secuencias del batch de golpe
            # Llamamos al método interno de la cRBM que creamos anteriormente
            batch_generated = rbm.generate_sequence(
                seed_frames=seed_batch, 
                n_steps_future=steps_to_predict, 
                gibbs_steps=gibbs_steps
            )
            
            generated_sequences.append(batch_generated.cpu().numpy())
            pbar.update(current_batch_size)
            
    pbar.close()
    
    matriz_final = np.concatenate(generated_sequences, axis=0)
    return matriz_final
