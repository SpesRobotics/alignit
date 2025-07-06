from datasets import load_from_disk
import transforms3d as t3d
from alignit.utils.zhou import sixd_se3
from alignit.utils.tfs import print_pose
import numpy as np

def main():
    index = 0

    dataset = load_from_disk("data/duck")
    image = dataset[index]["images"][0]
    action_sixd = np.array(dataset[index]["action"])
    action = sixd_se3(action_sixd)

    print_pose(action)
    image.show()


if __name__ == "__main__":
    main()
