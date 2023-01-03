class Dataframe:
    def __init__(self):
        self.rows = []
        self.columns = []
        self.label = ""
        self.classes = []

    def read_file(self, dataframe, file):
        f = open(file)
        data = f.read()
        f.close()
        lines = data.splitlines()
        dataframe.rows = [rows.split(',') for rows in lines]
        dataframe.columns = dataframe.rows.pop(0)
        dataframe.label = dataframe.columns.pop()

        for row in dataframe.rows:
            dataframe.classes.append(row.pop())
