# Pr_Automated_NP_Synthesis

All the optimization scripts will run in the *optimization* environment, and the pumps will run in the *environment* environment.

Do the following steps in order:

- First, think long and hard whether you really want to run the optimization and even longer and harder about the automation part. During optimization, you will have to do many 'weird' experiments, and during automation, you might often have to see the setup make 'stupid' mistakes. So, if you decide to go forward, be patient. Be kind. The setup was made with a lot of care and effort, treat it well, and it will be good to you too. :)

Now that you are sure that you need to automate, let's begin. There are tutorial videos in the 'tutorials' folder for your aid.

# Things to do before running the experiments

I am assuming you have made sure that the tubes are clean and all the channels are callibrated.
**Tip for cleaning the tubes with less difficulty:** Push the pasteur pipette or a syringe into one side of the tube. Thw whole tube will now *effectively* function as the pipette or the syringe. Dip one end into aqua regia or any other cleaning solution. You should be easily such the cleaning solution into the tube, and remove it. I have found it to be much more convenient than pushing the solution from one end. Try your luck with other cleaning procedures.
**Second Note for cleaning:** For the thin tubes, the syringe pump works much better. 
**Third Note:** I have also noticed that the tubes that we have, especially the ones with yellow stoppers, are not very easy to clean. I had to keep the aqua regia in the tubes for at least 15 mins to dissolve and remove all the reduced gold from the tubes. I used the concentrated aqua regia and even then had to keep for a long long time. Much better to procure chemical resistant tubes.  

Fill all the pump channels with desired stock solutions, and make sure each channel is primed. That is, all the tubes are filled completely with the desired stock solution. I usually flow 1mL of stock solutions in all the channels.

# Running the experiments

## Let's first clean the folders

- First, make sure that there is no old file in DATA_UV_DIR_PATH, src folder in both PREST and PREST_2 folder. Runn the clean-folders.py file to remove all the files from folders and save the files in archived-files folder with today's date.

## It is time to initialize the spectrophotometer

- Next, run the initialize_spectrophotometer.py folder. Here, the terminal will ask you to take the baseline. Make sure you fill the flow cell with reference solution before taking the baseline. 

- Next, the terminal will prompt you to take the dark spectrum.  Turn off the lamp and take the dark spectrum. *Do not forget to turn back on the lamp again.*

- Successful execution of this script will generate spectrophotometer_data.txt file in the src folder itself. This data will be used again and again during the iterations to calculate the absorbance of the sample. 

## Running the optimization + automation

- Just to repeat an important point, make sure that the pumps are callibrated, tubes are clean, and all channels are primed.

- Open two terminals in the *environment* environment. Thest two terminals are for the two pumps. Go to *hardware* > *prest* and run *main.py* file.

- This should open a serial control in the terminal itself, if the pumps are connected properly with the computer. 

- Go to the automatic mode, and now the pumps should be ready to execute the commands given to it by the optimization script. 

- Repeat this for the second pump too.

- Next, run the *script-control-multi-objective_peak-position_final.py* file. It is made in such a way that the setup will use peak position and absorbance ratios as the two objectives for the optimization experiment.

- Once all the experiments are over, run the *plot_spectrum.py* file, to generate the graphs.

- Finally, run the *move-file.py* to move all the data, including the parameters generated into data-uv-dir path in a new folder with today's date, and similarly, all the plots in resuls,figs,new folder with today's date. *Note that the terminal will prompt you for the name of the experiment. It will be used to name the folder.*