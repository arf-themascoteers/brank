from task_runner import TaskRunner

if __name__ == '__main__':
    tasks = {
        "algorithms" : ["ssprasenet","zhang"],
        "datasets" : ["indian_pines"],
        "target_sizes" : [5, 10, 15, 20, 25, 30]
    }
    ev = TaskRunner(tasks,1,10,"1.csv",skip_all_bands=True)
    ev.evaluate()