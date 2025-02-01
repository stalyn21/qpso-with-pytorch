import numpy as np
import matplotlib.pyplot as plt
import torch
import logging

def plot_cross_validation_losses(all_losses, dataset_name, optimizer):
    """
    Plotea las curvas de pérdida para cross-validation usando matplotlib con LaTeX.
    Genera dos gráficos separados: uno para las pérdidas y otro para la diferencia.

    Args:
        all_losses: Lista de diccionarios con las pérdidas de cada fold
        dataset_name: Nombre del dataset para el título
        optimizer: Nombre del optimizador usado
    """
    # Configurar LaTeX
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}"
    })

    def to_numpy(tensor_or_list):
        if isinstance(tensor_or_list, (list, np.ndarray)):
            return np.array([to_numpy(x) for x in tensor_or_list])
        elif isinstance(tensor_or_list, torch.Tensor):
            return tensor_or_list.cpu().detach().numpy()
        return tensor_or_list

    # Convertir pérdidas a arrays numpy
    try:
        train_losses = np.array([to_numpy(losses['train']) for losses in all_losses])
        val_losses = np.array([to_numpy(losses['val']) for losses in all_losses])
        logging.info("Train plotting shape: %s", train_losses.shape)
        logging.info("Val plotting shape: %s", val_losses.shape)
    except Exception as e:
        logging.error(f"Error al convertir las pérdidas: {e}")
        logging.info(f"Tipo de all_losses: {type(all_losses)}")
        logging.info(f"Contenido de all_losses[0]: {all_losses[0]}")
        raise

    epochs = range(len(train_losses[0]))

    # Calcular estadísticas
    train_mean = np.mean(train_losses, axis=0)
    train_std = np.std(train_losses, axis=0)
    val_mean = np.mean(val_losses, axis=0)
    val_std = np.std(val_losses, axis=0)

    # Definir colores
    train_color = 'm'
    val_color = 'c'
    diff_color = 'purple'

    # -------------------- Primer gráfico: Pérdidas --------------------
    plt.figure(figsize=(12, 6))

    # Plotear curvas individuales
    for fold in range(len(all_losses)):
        plt.plot(epochs, train_losses[fold],
                color=train_color,
                alpha=0.15,
                linestyle='-')
        plt.plot(epochs, val_losses[fold],
                color=val_color,
                alpha=0.15,
                linestyle='-')

    # Plotear medias
    plt.plot(epochs, train_mean,
             label=r'$\mathcal{T}_{\mathrm{train}}$ (mean)',
             color=train_color,
             linewidth=2)
    plt.plot(epochs, val_mean,
             label=r'$\mathcal{T}_{\mathrm{val}}$ (mean)',
             color=val_color,
             linewidth=2)

    # Añadir intervalos de confianza
    plt.fill_between(epochs,
                    train_mean - train_std,
                    train_mean + train_std,
                    color=train_color,
                    alpha=0.2)
    plt.fill_between(epochs,
                    val_mean - val_std,
                    val_mean + val_std,
                    color=val_color,
                    alpha=0.2)

    # Añadir texto informativo
    info_text = f'Number of folds: {len(all_losses)}\n'
    info_text += f'Final train loss: {train_mean[-1]:.4f} ± {train_std[-1]:.4f}\n'
    info_text += f'Final val loss: {val_mean[-1]:.4f} ± {val_std[-1]:.4f}'

    plt.text(0.02, 0.98, info_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round',
                      facecolor='white',
                      alpha=0.8))

    # plt.title(r'\textbf{Mean Validation and Training Losses - ' + dataset_name + r' Dataset}',
              # pad=20)
    plt.xlabel(r'\textbf{Epoch ($\mathcal{E}$)}')
    plt.ylabel(r'\textbf{Loss ($\mathcal{L}$)}')
    plt.grid(False)
    plt.legend(loc='upper right',
              frameon=True,
              fancybox=True,
              shadow=True,
              framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f'./metrics/graphics/loss/{optimizer}/{dataset_name}_losses_comparison_{optimizer}.png',
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.1)
    plt.close()

    # -------------------- Segundo gráfico: Diferencia --------------------
    plt.figure(figsize=(12, 6))

    diff_mean = val_mean - train_mean
    diff_std = np.sqrt(val_std**2 + train_std**2)

    plt.plot(epochs, diff_mean,
            color=diff_color,
            linewidth=2,
            label=r'$\mathcal{T}_{\mathrm{val}} - \mathcal{T}_{\mathrm{train}}$')
    plt.fill_between(epochs,
                    diff_mean - diff_std,
                    diff_mean + diff_std,
                    color=diff_color,
                    alpha=0.2)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)

    # plt.title(r'\textbf{Training-Validation Loss Difference - ' + dataset_name + r' Dataset}',
              # pad=20)
    plt.xlabel(r'\textbf{Epoch ($\mathcal{E}$)}')
    plt.ylabel(r'\textbf{Loss Difference ($\Delta\mathcal{L}$)}')
    plt.grid(False)
    plt.legend(loc='upper right',
              frameon=True,
              fancybox=True,
              shadow=True,
              framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f'./metrics/graphics/loss/{optimizer}/{dataset_name}_loss_difference_{optimizer}.png',
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.1)
    plt.close()