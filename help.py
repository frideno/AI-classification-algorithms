class DataHandler:
    """
    class for handling data set
    read data file, write answers and so on.

    """
    def __init__(self):
        self.attributes_params = {}
        self.values = []
        self.classes = []

    def load_metadata(self, data_file_name):

        attributes = {}
        classes = set()
        with open(data_file_name) as f:
            # read attribute names + order
            attributes_order = f.readline()[:-1].split('	')
            for a in attributes_order:
                attributes[a] = set()

            # read data
            for line in f:
                splt = line[:-1].split('	')
                for att, val in zip(attributes_order, splt):
                    attributes[att].add(val)
                classes.add(splt[-1])

            # set attributes names and values.
            for att_name, param_names in attributes.items():
                self.attributes_params[att_name] = sorted(list(param_names))

            # make last attribute the classes:
            self.classes = sorted(list(classes))

    def load_data(self, data_file_name):


        """
        extracting data from filename, by seperator.
        attributes turn into numerical values by attributes file.
        """

        # read data from data file by attributes read.
        with open(data_file_name) as f:

            # read attribute names + order
            attributes_order = f.readline()[:-1].split('	')
            # read data
            for line in f:
                splt = line[:-1].split('	')
                vl_numerical = [self.attributes_params[attributes_order[i]].index(splt[i]) for i in range(len(splt))]
                self.values.append(vl_numerical)

        # print('extracted ' + str(len(self.values) )+ ' values from ' + data_file_name)



def hamming_distance(arr1, arr2):
    """
    :return hamming distance of two given lists arr1, arr2 (assumes equal length).
    """
    diffs = 0
    for c1,c2 in zip(arr1, arr2):
        if c1 != c2:
            diffs += 1
    return diffs


# helpful functions from numpy:
def argmax(arr):
    return arr.index(max(arr))