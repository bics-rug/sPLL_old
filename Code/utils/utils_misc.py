import os
def what_time_is_it():
    import time
    from datetime import datetime
    a = datetime.fromtimestamp(time.time())
    now = a.strftime("%d_%h_%H_%M")
    return now
def create_folder(folder):
    import os
    try:
        os.mkdir(folder)
    except FileExistsError:
        while 1 == 0:
            print('we')
    return folder

def create_image_folder(IMAGE_PATH, code_name, parameters):
    if parameters['save_plots'] == True:
        my_time = what_time_is_it()
        parameters['Images_Folder'] = os.path.join(IMAGE_PATH, str(my_time + code_name))
        create_folder(parameters['Images_Folder'])
        return parameters

def update_progress(progress,string):
    import sys
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\r" + string + ": [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), round(progress,2)*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()
