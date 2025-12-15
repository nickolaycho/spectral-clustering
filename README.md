# spectral-clustering
an implementation of the spectral clustering algorithm

REQUIREMENTS
	- A code editor like Visual Studio Code
	- Python 3.8.10 (https://www.python.org/downloads/release/python-3810/)

INSTRUCTIONS

1. Create a virtual environment with required libraries
	NOTE: IF YOU ALREADY CREATED A VIRTUAL ENVIRNOMENT FOR THE FREE HOMEWORK, SKIP TO STEP 2

	1a) Open a terminal or command prompt and navigate to the directory containing this repository on your PC
	1b) Once you are there, create a virtual environment by typing the following command:
			python -m venv nameOfYourEnv
	1c) Activate the just created virtual environment by typing in the terminal:
		- For Windows: <env_name>\Scripts\activate.bat 
		- For Unix/Linux: source <env_name>/bin/activate 
	1d) Install the required libraries by running the following command:
	pip install -r "/[path to requirements]/requirements.txt"
	[note that the requirements.txt file is in the spectral_clustering folder]
	1e) When the installations are completed, you may close the terminal.
	
2. In a code editor like VisualStudio, go on "File", then "Open folder" and browse to the location where you downloaded this repository and select the spectral_clustering folder, then hit "open" to open it.

3. Select the Python interpreter for the project:
	Open the command palette in your code editor (usually accessible through Ctrl + Shift + P).
	Search for "Python: Select Interpreter" and choose it.
	From the list of available interpreters, select the virtual environment you created in step 1. This will ensure that the project uses the correct Python version and libraries. If you do not see it in the list, navigate to the folder where you installed the virtual environment, then "bin", then select "python" as the interpreter.

4. The files:
	- deflation_method.py
	- inverse_power_method.py
	- utils.py
contain useful functions that have been used in the spectral clustering implementation (see report for details about each one) and the functions used to compute the smallest eigenvalues of a symmetric matrix). 

5. The file:
	- spectral_clustering_code.py
when runned, outputs in the folder "figs" the graphs used in the report, with an extension indicating the timestamp. I left the folder "figs" empty, so when you will run this file you will get the figures there. 

Feel free to explore the code and modify it according to your needs. If you have any questions or need further clarification, please reach out. Note: in line 57 of spectral_clustering_code.py, you can change the dataset where to perform the analysis (circle_df or spiral_df).
