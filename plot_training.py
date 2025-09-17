# plot_env.py
import argparse, os
import numpy as np
import matplotlib.pyplot as plt

def load_env_data(env_dir):
    inner_path = os.path.join(env_dir, "inner_avg_steps.npy")
    lang_path  = os.path.join(env_dir, "lang_avg_steps.npy")
    if not (os.path.exists(inner_path) and os.path.exists(lang_path)):
        raise FileNotFoundError(f"Missing npy files in {env_dir}")
    inner = np.load(inner_path)
    lang  = np.load(lang_path)
    env_name = os.path.basename(os.path.normpath(env_dir))
    return inner, lang, env_name
def plot_line(env_dir, save_path=None, show=True, dpi=300, out_dir="figures"):
    inner, lang, env_name = load_env_data(env_dir)

    plt.plot(inner, label="MAML Policy")
    plt.plot(lang,  label="Lang-adapted Policy")
    plt.xlabel("Episodes")
    plt.ylabel("Average Steps")
    plt.title(env_name)
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()

    # Decide where to save
    if save_path is None:
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"{env_name}.png")

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"Saved {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot MAML vs Lang steps for one environment")
    parser.add_argument("env_dir", help="Path to environment folder (e.g., metrics/GoToLocal)")
    parser.add_argument("--save", metavar="FILE", help="Save figure to file (PNG, PDF, etc.)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figure")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot window")
    args = parser.parse_args()

    plot_line(args.env_dir, save_path=args.save, show=not args.no_show, dpi=args.dpi)