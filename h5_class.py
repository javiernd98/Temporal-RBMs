import numpy as np
import torch
import h5py
import time
import matplotlib.pyplot as plt
import os
import glob


def save_checkpoint_h5(filename, rbm, step, train_energy=None, test_energy=None):
    #"""
    #Guarda checkpoints en H5. 
    #- Si 'step' es entero -> guarda en grupo 'step_N'
    #- Si 'step' es string -> guarda en grupo con ese nombre (ej: 'final_model')
    #"""
    with h5py.File(filename, 'a') as f:
        # Determinar nombre del grupo
        group_name = f'step_{step}' if isinstance(step, int) else str(step)
    
        # Limpieza por si re-ejecutamos
        if group_name in f: 
            del f[group_name]
        
        g = f.create_group(group_name)
    
        # Guardamos tensores (siempre pasar a CPU y Numpy)
        g.create_dataset('W', data=rbm.W.cpu().detach().numpy())
        g.create_dataset('vbias', data=rbm.vbias.cpu().detach().numpy())
        g.create_dataset('hbias', data=rbm.hbias.cpu().detach().numpy())

        # NUEVO: Guardar matrices temporales A y B si existen en el modelo
        if hasattr(rbm, 'A'):
            g.create_dataset('A', data=rbm.A.cpu().detach().numpy())
        if hasattr(rbm, 'B'):
            g.create_dataset('B', data=rbm.B.cpu().detach().numpy())


        
        # Metadatos útiles
        g.attrs['timestamp'] = time.time()
        if train_energy is not None: g.attrs['train_energy'] = train_energy
        if test_energy is not None: g.attrs['test_energy'] = test_energy
    
        # Si es un paso numérico, lo guardamos también como atributo para ordenar fácil luego
        if isinstance(step, int):
            g.attrs['step_index'] = step

    # Feedback en consola solo para hitos importantes o final
    #if not isinstance(step, int) or step % 5000 == 0:
        #print(f"   💾 Checkpoint H5 guardado: {group_name}")
