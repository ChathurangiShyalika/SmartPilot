## Process Ontology for Future Factories Assembly Cell


### Installing Docker
* Download the most recent version of docker as per your OS from here - https://docs.docker.com/desktop/release-notes/


### Steps to run
* Change the $USER and $UID in .env file. To find out the values `echo $USER` and `echo $UID` from your command terminal
* Create two folders insider the project folder 
* * A folder to save the graph database (graph_data)
* * A folder where the original raw data files are stored (This doesn't exist for now. So create an empty folder named raw_files)
* Mention these paths in the docker-compose.yml file. Mention full path.
* * `<path to raw_folder>:/import`
* * `<path to graph_data>:/data`
* Download the plugin [apoc-5.16.0-extended.jar](https://drive.google.com/file/d/12iVJVKnC4H-dYCx_-vhaKJwk9zzpXWzy/view?usp=sharing) and put it into a folder named `plugins` mention full path in the docker-compose file. Better to put this somewhere outside the project folder. 
* * `<path to plugins>:/var/lib/neo4j/plugins`
* Command to stand up the docker container `docker compose up --build`
* Access the notebook and neo4j from ports `https://localhost:8888` and `https://localhost:7474` respectively. This is just a step if you need to visualize something. So it can be skipped.
* Username and password for neo4j is in .env file
* Password for Jupyterlab is in .env file
* If you want to execute the python scripts, check the isntruction under that section
* Command to get the docker container down `docker compose down -v`
* NOTE: Currently, jupyterlab is commented out in the docker-compose.yml. Uncomment it before starting the docker is needed


### Python Scripts
* The python files are present in the folder python_files
* Create a virtual environment using `python3 -m venv <env_name> `. Install the required packages using `pip install -r requirements.txt`
* Activate the virutal environment before you have to run the script.
* The script main.py has the template to input your data and get the explanation behind anomalies
* The sample input data files are in mfg-data/anomaly_data. Make sure the column names are exactly the same if you plan on using a new file.
* The sample output files are in mfg-data/results


### ERRORS
* if `docker command not found` after the installation, add it to your path. For mac, ` export PATH="$PATH:/Applications/Docker.app/Contents/Resources/bin/" `. Link - https://stackoverflow.com/questions/64009138/docker-command-not-found-when-running-on-mac 

* if `OSError:No space left on deivce` while running docker compose, do `docker system prune -af`. Link - https://stackoverflow.com/questions/44664900/oserror-errno-28-no-space-left-on-device-docker-but-i-have-space 