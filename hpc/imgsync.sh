#!/bin/bash



while true; do
    # Check if imgsync.png file exists
    echo 'trying rsync...'
    rsync -avz -e "ssh -i '/Users/aleksygalkowski/Documents/Projects/ucph/social-data-science/2023 sem 3/aml-itu/aml_project_2023/hpc/utils/ssh'" agac@hpc.itu.dk:/home/data_shares/fear_sds/hpc/plots/ "/Users/aleksygalkowski/Documents/Projects/ucph/social-data-science/2023 sem 3/aml-itu/aml_project_2023/hpc/utils/plots/"

    # Sleep for X seconds
    sleep 5
done

feh utils/plots/loss_plot.png
feh --reload 5 utils/plots/loss_plot.png

# ssh -i "/Users/aleksygalkowski/Documents/Projects/ucph/social-data-science/2023 sem 3/aml-itu/aml_project_2023/hpc/utils/plots/id_rsa.pub" agac@hpc.itu.dk



