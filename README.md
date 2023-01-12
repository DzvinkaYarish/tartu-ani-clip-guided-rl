# CLIP-Guided RL

## Installation
```bash
conda create --name ani python=3.8
conda install -c conda-forge jupyterlab
conda install -c anaconda ipywidgets
conda install -c anaconda scipy
conda install -c conda-forge swig
pip install Cython
pip install open_clip_torch
pip install stable-baselines3[extra]
pip install pyglet==1.5.27
pip install tensorboardX
pip install gym[all]
```

For *Windows*, you will also need to install [Mujoco](https://www.roboti.us/index.html):
- Install [mjpro150](https://www.roboti.us/download.html) to `C:\Users\<User>\.mujoco\<mjpro150>`
- Install [license key](https://www.roboti.us/license.html) to `C:\Users\<User>\.mujoco\mjkey.txt`
- Add `C:\Users\<User>\.mujoco\mjpro150\bin` to PATH
- Enable Win32 long paths
- run `pip install gym[all]` again

## Experiments

### Models
 - [CLIP](https://github.com/mlfoundations/open_clip)
 - [CLOOB](https://github.com/ml-jku/cloob)

### Environments
- [CartPole](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)
- [LunarLander](https://www.gymlibrary.dev/environments/box2d/lunar_lander/)

## Colab

### Gym Installation Issues
Sometimes, OpenAI Gym installation fails on Colab. 
Therefore, it is recommended to `pip install gym` first, before all other packages

```commandline
!pip install gym[box2d]
!pip install stable-baselines3[extra]
!pip install open_clip_torch
!pip install pyglet==1.5.27
!pip install tensorboardX
!pip install ftfy
```

### Gym Render Issues
In order to render OpenAI Gym in Colab, 
please follow these [instructions](https://stackoverflow.com/questions/50107530/how-to-render-openai-gym-in-google-colab)

```commandline
!sudo apt-get update
!sudo apt-get install x11-utils 
!sudo pip install pyglet 
!sudo apt-get install -y xvfb python-opengl
!pip install pyvirtualdisplay
```

```python
from pyvirtualdisplay import Display
display = Display(visible=0, size=(400, 300))
display.start()
```

### CLOOB

CLOOB requires additional Python modules to work.

```commandline
!git clone https://github.com/DzvinkaYarish/tartu-ani-clip-guided-rl.git
!cp -r /content/tartu-ani-clip-guided-rl/cloob /content
```
