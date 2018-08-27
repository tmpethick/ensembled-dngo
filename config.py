import sys
import os
import uuid


def get_config(top_folder = "outputs"):
    database = os.path.join(top_folder, "entries.csv")
    remote_temp_database = os.path.join(top_folder, "entries.remote.csv")
    return dict(
        top_folder=top_folder, 
        database=database,
        remote_temp_database=remote_temp_database)


def get_model_config(uid, model_shortname, config):
    folder = os.path.join(config['top_folder'], model_shortname)
    plot_folder = os.path.join(folder, "plots")

    command_path = os.path.join(folder, "command.txt")
    obs_X_path = os.path.join(folder, "obs-X.npy")
    obs_Y_path = os.path.join(folder, "obs-Y.npy")
    regret_plot_path = os.path.join(folder, "immediate_regret.png")

    return dict(
        folder=folder,
        plot_folder=plot_folder,
        command_path=command_path,
        obs_X_path=obs_X_path,
        obs_Y_path=obs_Y_path,
        regret_plot_path=regret_plot_path,
    )
