# Generative Human Motion Stylization in Latent Space, ICLR 2024
### [[Project Page]](https://yxmu.foo/GenMoStyle/) [[Paper]](https://openreview.net/pdf?id=daEqXJ0yZo) [[Dataset]](https://drive.google.com/drive/u/1/folders/1Cnc0n8GhDrqjcP68_j5xb6qRx72aQXWX)[[Checkpoints]](https://drive.google.com/file/d/1yTDhGbD58-zmul33iA31fmtdYoJNqfil/view?usp=share_link)
![teaser_image](./assets/teaser.png)
The release of the training codes will be delayed due to company review requirements.

## TODO
- [ ] Add evaluation codes
- [ ] Add training codes
- [x] Add evaluation codes
- [x] Release checkpoints
- [x] Release processed dataset
- [x] Add codes for dataset
- [x] Add baseline implementations

  
## Baseline Implementations
The baseline models are implemented in the subfolders (i.e. `./baseline/unpaired_motion`, `./baseline/diverse_stylize`, `./baseline/motion_puzzle`), built from their offical implementations on github. For more details, please refer to the official repositories [Aberman et al.](https://github.com/DeepMotionEditing/deep-motion-editing), [Park et al.](https://github.com/soomean/Diverse-Motion-Stylization), [Jang et al.](https://github.com/DK-Jang/motion_puzzle).

### Baseline Scripts
All training and testing scripts are documented in `./$baseline_path/eval_scripts.txt`. 

### Misc
Contact ymu3@ualberta.ca for further questions. A rough codebase for our method could be found through our OpenReview page ([Download](https://openreview.net/attachment?id=daEqXJ0yZo&name=supplementary_material)).

## Acknowledgement
We have intensively borrow codes from the following repositories. Many thanks to the authors for sharing their codes.
- [deep-motion-editing](https://github.com/DeepMotionEditing/deep-motion-editing)
- [Diverse-Motion-Stylization](https://github.com/soomean/Diverse-Motion-Stylization)
- [Motion-puzzle](https://github.com/DK-Jang/motion_puzzle)