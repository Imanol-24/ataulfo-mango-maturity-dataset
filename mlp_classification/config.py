import argparse


DEFAULT_R01 = "ruta/a/R01.csv"
DEFAULT_R02 = "ruta/a/R02.csv"
DEFAULT_R03 = "ruta/a/R03.csv"
DEFAULT_R04 = "ruta/a/R04.csv"


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file_r01", default=DEFAULT_R01)
    parser.add_argument("--file_r02", default=DEFAULT_R02)
    parser.add_argument("--file_r03", default=DEFAULT_R03)
    parser.add_argument("--file_r04", default=DEFAULT_R04)

    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)

    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr_list", default="0.003,0.005")
    parser.add_argument("--smooth_list", default="0.02,0.015,0.01")
    parser.add_argument("--l2_list", default="1e-5,3e-5,1e-4")
    parser.add_argument("--w1", type=float, default=1.2)
    parser.add_argument("--w2_list", default="1.2,1.3,1.4,1.5")
    parser.add_argument("--w3_list", default="1.2,1.3,1.4,1.5")
    parser.add_argument("--bn_list", default="true,false")

    parser.add_argument("--epochs", type=int, default=120)

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Si se activa, reanuda el sweep usando clasico_sweep_results.csv",
    )
    parser.add_argument("--results_csv", default="clasico_sweep_results.csv")

    return parser