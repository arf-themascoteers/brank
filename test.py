from task_runner import TaskRunner

if __name__ == '__main__':
    tasks = {
        "algorithms" : ["linspacer","bsnet","bsdr","rec"],
        "datasets" : ["indian_pines","paviaU","salinasA"],
        "target_sizes" : [5, 10, 15, 20, 25, 30]
    }
    ev = TaskRunner(tasks,1,1,"1.csv")
    ev.evaluate()