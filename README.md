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

### CLIP

CLIP requires additional Python modules to work.

```commandline
!git clone https://github.com/DzvinkaYarish/tartu-ani-clip-guided-rl.git
!cp -r /content/tartu-ani-clip-guided-rl/utils.py /content
```

Weights can be found [here](https://github.com/mlfoundations/open_clip/releases/tag/v0.2-weights). 
CLIP will download them automatically

### CLOOB

CLOOB requires additional Python modules to work.

```commandline
!git clone https://github.com/DzvinkaYarish/tartu-ani-clip-guided-rl.git
!cp -r /content/tartu-ani-clip-guided-rl/utils.py /content
!cp -r /content/tartu-ani-clip-guided-rl/cloob /content
```

Weights can be found [here](https://ml.jku.at/research/CLOOB/downloads/checkpoints/). 
You have to download them manually.

```commandline
!mkdir checkpoints
!wget https://ml.jku.at/research/CLOOB/downloads/checkpoints/cloob_rn50_yfcc_epoch_28.pt
!mv /content/cloob_rn50_yfcc_epoch_28.pt /content/checkpoints/cloob_rn50_yfcc_epoch_28.pt
```

### How to modify Gym environment by populating it with images

Gym Box2D environments are quite minimalistic, and it is unlikely CLIP has seen images like that during training. 
This minimalism might confuse CLIP which as a result will produce incorrect similarity scores. 
Generally, creating prompts for such environments might be tricky which 
led to the idea of modifying the environment by pasting images of more realistic 
objects that CLIP might have seen during training. 
For example, in Lunar Lander you can replace the spaceship with an image of the cat 
which hopefully will make the job easier for CLIP. 
However, in order to do that you have to modify the source code of the environment. 
Below we provide the instructions on how to do that.

1. Before we start modifying the environment, we need to make some minor changes 
inside `rendering.py` ("\<venv\>/envs/ani/Lib/site-packages/gym/envs/classic_control/rendering.py")
2. Find the `Viewer` class and add the following method
```python
    def draw_image(self, filename, width, height):
        geom = Image(filename, width, height)
        self.add_onetime(geom)
        return geom
```
3. Open the desired environment code (e.g. "\<venv\>/envs/ani/Lib/site-packages/gym/envs/box2d/lunar_lander.py")
4. We only need to change the `render` method since that is what returns the image
5. Just before the return draw the necessary images with the following code.
```python
t = rendering.Transform(
    translation=(x, y),
    rotation=angle,
)

self.viewer.draw_image(
    "<image_path>",
    width, height
).add_attr(t)
```
Here we draw the image and apply transformation `t` that moves the 
image to coordinates `x` and `y` and then rotates the image by `angle` (radians).
6. A complete example for Lunar Lander that replaces the shape-ship with a cat can be found below. 
You need to replace the `render` method with the following code.
```python
    def render(self, mode="human"):
        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W / SCALE, 0, VIEWPORT_H / SCALE)

        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (
                max(0.2, 0.2 + obj.ttl),
                max(0.2, 0.5 * obj.ttl),
                max(0.2, 0.5 * obj.ttl),
            )
            obj.color2 = (
                max(0.2, 0.2 + obj.ttl),
                max(0.2, 0.5 * obj.ttl),
                max(0.2, 0.5 * obj.ttl),
            )

        self._clean_particles(False)

        for p in self.sky_polys:
            self.viewer.draw_polygon(p, color=(0, 0, 0))

        ship = None
        ship_trans = None
        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(
                        f.shape.radius, 20, color=obj.color1
                    ).add_attr(t)
                    self.viewer.draw_circle(
                        f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2
                    ).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    if ship is None:
                        ship = path
                        ship_trans = trans
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        for x in [self.helipad_x1, self.helipad_x2]:
            flagy1 = self.helipad_y
            flagy2 = flagy1 + 50 / SCALE
            self.viewer.draw_polyline([(x, flagy1), (x, flagy2)], color=(1, 1, 1))
            self.viewer.draw_polygon(
                [
                    (x, flagy2),
                    (x, flagy2 - 10 / SCALE),
                    (x + 25 / SCALE, flagy2 - 5 / SCALE),
                ],
                color=(0.8, 0.8, 0),
            )

        x, y = centroid(ship)
        height = 3.4
        width = 3.4

        t = rendering.Transform(
            translation=(x, y),
            rotation=ship_trans.angle + 0.34,
        )

        self.viewer.draw_image(
            "<image_path>",
            width, height
        ).add_attr(t)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    @staticmethod
    def centroid(vertexes):
        _x_list = [vertex[0] for vertex in vertexes]
        _y_list = [vertex[1] for vertex in vertexes]
        _len = len(vertexes)
        _x = sum(_x_list) / _len
        _y = sum(_y_list) / _len
        return _x, _y
```
