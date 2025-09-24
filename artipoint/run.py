import hydra
from omegaconf import DictConfig

from artipoint.track.artipoint import ArtiPoint


@hydra.main(version_base=None, config_path="configs", config_name="artipoint")
def main(cfg: DictConfig) -> None:
    runner = ArtiPoint(cfg)
    runner.run()


if __name__ == "__main__":
    main()
