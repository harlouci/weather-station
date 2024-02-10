"""
Launches a long running process as an Agent to run flows,
expected to be ran in docker compose.

The intended goal is for the docker (compose) container
to launch this file as the main process (in "command").

Running this script will create a long running process
.serve().

"""

import prefect
#from weather.pipelines.flows.full_pipeline import automated_pipeline
from weather.pipelines.flows.automated_pipeline import automated_pipeline

if __name__ == "__main__":
    # NOTE(Participant): This will make Prefect available to run the deployments
    #                    (this local process will act as an agent and launch
    #                    runs when asked)

    # To launch: prefect deployment run --param max_runs=5  automated-pipeline/long-serving
    # deploy_automated = automated_pipeline.to_deployment(name="long-serving", cron="*/10 * * * *") # IBAN
    deploy_automated = automated_pipeline.to_deployment(name="local")   # IBAN

    print("Flows have been deployed, also serving the flows on the local machine using an agent")
    prefect.serve(deploy_automated)
