# REINVENT4Surfactants
by Hannes Öhman 

 MSc. Complex Adaptive Systems & BSc. Chemical Enginnering with Engineering Physics 

at Chalmers University of Technology

## Installation:
1. Start by installing REINVENT4 in a seperate folder from this project.

Reinvent can installed from the official github:
[REINVENT4](https://github.com/MolecularAI/REINVENT4)

2. Clone this repository into your desired folder.
3. Create a virtual environment for this project.
4. Install REINVENT into that virutal environment using the steps described in the github.
5. Copy (or create symbolic links for) the scoring functions into the components folder of REINVENT in your virtual environment.

    Example:
    ```
    venv/lib/python3.13/site-packages/reinvent_plugins/components
    ```

## How to use:
Edit the *config.json* file contained in the main folder. Here you can set the desired parameters for your run.

+ **WORKFLOW_NAME:** The version folder of the framework.
+ **GROUP_NAME:** A name you can set to more easily organize runs into groups, for example if they all belong to a single project.
+ **RUN_NAME:** The name of run you are about to do. This will be found inside the **GROUP_NAME** folder and will have a timestamp at the end of the folder name.