def plot_flat_metric_reference(save_path='fig_flat_metric.png'):
    r = np.linspace(1, 10, 100)
    flat = np.ones_like(r)

    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(r, flat, label=r'$g_{tt}(r) = 1$', linestyle='--', color='gray')
    plt.xlabel(r'$r$', fontsize=12)
    plt.ylabel(r'$g_{tt}(r)$', fontsize=12)
    plt.title('Flat Minkowski Time Component', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

