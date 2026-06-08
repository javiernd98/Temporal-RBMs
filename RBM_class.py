import numpy as np
import torch
import os
import glob


class RBM_PCD_Energy:
    def __init__(self, n_vis_total: int, n_hid: int, n_stack: int, learning_rate=0.001, momentum=0.5, weight_decay=0.0001, device="cpu"):
        self.device = torch.device(device)
        self.n_vis = n_vis_total
        self.n_hid = n_hid
        self.lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.n_stack = n_stack
        self.frame_size = n_vis_total // n_stack

        self.W = torch.randn(n_vis_total, n_hid, device=self.device) * 0.01
        self.vbias = torch.zeros(n_vis_total, device=self.device)
        self.hbias = torch.zeros(n_hid, device=self.device)

        self.persistent_v = None

        self.vW = torch.zeros_like(self.W)
        self.vvbias = torch.zeros_like(self.vbias)
        self.vhbias = torch.zeros_like(self.hbias)

    def init_biases_statistics(self, data):
        # data puede venir de CPU o GPU, no importa
        p = torch.mean(data, dim=0)
        eps = 1e-5
        p = torch.clamp(p, eps, 1.0 - eps)
        
        # Calculamos el bias inicial
        initial_bias = torch.log(p / (1.0 - p))
        
        # --- CORRECCIÓN IMPRESCINDIBLE ---
        # Forzamos a que el bias se guarde en la GPU (self.device)
        # aunque los datos originales vinieran de la CPU.
        self.vbias = initial_bias.to(self.device) 
        
        print(f" Biases inicializados estadísticamente en {self.vbias.device}.")

    def _sigmoid(self, x):
        return torch.sigmoid(x)

    def free_energy(self, v):
        vbias_term = torch.matmul(v, self.vbias)
        wx_b = torch.matmul(v, self.W) + self.hbias
        hidden_term = torch.sum(torch.nn.functional.softplus(wx_b), dim=1)
        return -hidden_term - vbias_term

    def sample_h(self, v):
        mh = self._sigmoid(self.hbias + torch.matmul(v, self.W))
        return torch.bernoulli(mh), mh

    def sample_v(self, h):
        mv = self._sigmoid(self.vbias + torch.matmul(h, self.W.t()))
        return torch.bernoulli(mv), mv

    def train_step_pcd(self, v_data, k=1):
        batch_size = v_data.shape[0]
        _, mh_data = self.sample_h(v_data)

        if self.persistent_v is None or self.persistent_v.shape[0] != batch_size:
            self.persistent_v = v_data.clone()

        v_chain = self.persistent_v.detach() ## solo nos quedamos con el último v_chain del update anterior y nos olvidamos del resto 

        for _ in range(k):
            h_chain, _ = self.sample_h(v_chain)
            v_chain, _ = self.sample_v(h_chain)

        _, mh_chain = self.sample_h(v_chain)
        self.persistent_v = v_chain

        ## Rao-Blackwell en la utilización al computar el gradiente de la probabilidad de activación de los hidden nodes y no los valores binarios
        ## Esto representa mucho mejor la realidad, en cierto modo nos deshacemos de la elección probabilística de Bernouilli
        ## El cálculo del gradiente que antes se encontraba dentro de _init_params de Nico ahora se encuentra aquí
        
        pos_grad = torch.matmul(v_data.t(), mh_data)
        neg_grad = torch.matmul(v_chain.t(), mh_chain)

        ## Calculamos los gradientes 

        grad_W = (pos_grad - neg_grad) / batch_size - (self.weight_decay * self.W)
        grad_vbias = (v_data.sum(0) - v_chain.sum(0)) / batch_size
        grad_hbias = (mh_data.sum(0) - mh_chain.sum(0)) / batch_size

        ## aquí la aplicación del gradiente la hacemos ligeramente distinta, no sólo multiplicamos lr*grad, si no que introducimos un momento de inercia
        ## que lo que hace es recompensar el movimiento en direcciones que parecen haber reducido la velocidad más de un paso seguido
        ## esto también ayuda a estabilizar en caso de que nos encontremos en un vaye estrecho
        
        
        self.vW = self.momentum * self.vW + self.lr * grad_W  ## self.vW es el acumulador de gradientes pasados, nos ayuda a tener inercia
        self.vvbias = self.momentum * self.vvbias + self.lr * grad_vbias
        self.vhbias = self.momentum * self.vhbias + self.lr * grad_hbias

        self.W += self.vW
        self.vbias += self.vvbias
        self.hbias += self.vhbias

        return torch.mean(self.free_energy(v_data)).item()

    def predict_next_frame(self, current_stack, gibbs_steps=20):
        # Predicción Condicional
        ## Aquí la diferencia entre samplear partiendo del puro ruido se encuentra en lo que metamos en current_stack
        ## Si le metemos  -- current_stack = torch.bernoulli(torch.rand((N_VIS_TOTAL), device=device)) -- tendremos random
        ## Si le metemos  -- current_stack = data[random_idx].clone() -- tendremos de inicio frames del dataset
        current_stack = current_stack.to(self.device)
        input_for_prediction = current_stack[self.frame_size:] ## aquí coge el stack de frames y quita el primero (recortamos los primeros 144 elementos del vector en este caso)
        known_size = self.frame_size * (self.n_stack - 1) ## fija los píxeles que conocemos y no queremos cambiar (todos menos los últimos 144 que se han liberado)
        
        joint_v = torch.zeros(self.n_vis, device=self.device) ## Crea un vector nuevo "vacío"
        joint_v[:known_size] = input_for_prediction ## Introduce los datos que queremos fijar
        joint_v[known_size:] = torch.rand(self.frame_size, device=self.device) ## Mete ruido al final

        for i in range(gibbs_steps):
            h, _ = self.sample_h(joint_v)
            v_sampled, mv = self.sample_v(h)
            
            final_output = v_sampled
            final_output[:known_size] = input_for_prediction # Clampeo estricto, aquí conserva los que teníamos originalmente y deja sólo el sampling del último frame
            joint_v = final_output

        return joint_v
