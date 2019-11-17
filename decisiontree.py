import math
import sys


def select_best_attribute(data, attributes, target):  # method to select the best attribute to start a node
    best_attribute = attributes[0]
    max_information_gain = 0

    for attribute in attributes:  # checking for the attribute with highest information gain

        if attribute != target:
            # calling method to calculate information gain of every attribute
            new_information_gain = information_gain(data, attributes, attribute, target)

        if new_information_gain > max_information_gain:   # checking for the highest information gain iteratively
            max_information_gain = new_information_gain
            best_attribute = attribute

    # Returning the attribute with the highest information gain
    return best_attribute


# method for calculating information gain of an attribute
def information_gain(data, attributes, attribute, target):
    frequency = {}
    subset_entropy = 0.0

    # Get the index of the attribute for which gain needs to be calculated
    index = attributes.index(attribute)  # position of the attribute in the attributes array

    # Calculating the occurrence of unique values of classification in the target array using dictionary form
    for entry in data:
        if entry[index] in frequency:
            frequency[entry[index]] += 1
        else:
            frequency[entry[index]] = 1

    for key in frequency.keys():    # calculating entropy of the attributes

        probability = frequency[key] / sum(frequency.values())

        data_subset = []
        for entry in data:
            if entry[index] == key:
                data_subset.append(entry)

        entropy = calculate_entropy(data_subset, attributes, target)
        subset_entropy += probability * entropy

    # information gain = system entropy - attribute entropy
    information_gain = calculate_entropy(data, attributes, target) - subset_entropy

    return information_gain


# Method to calculate entropy
def calculate_entropy(data, attributes, target):
    target_frequency = {}
    entropy = 0.0

    index = attributes.index(target)   # position of target attribute

    # Find the frequency of each target value
    for row in data:
        if row[index] in target_frequency:
            target_frequency[row[index]] += 1
        else:
            target_frequency[row[index]] = 1

    total_data_rows = sum(target_frequency.values())   # calculating entropy
    for unique_value in target_frequency:
        value_probability = target_frequency[unique_value] / total_data_rows
        entropy -= value_probability * (math.log(value_probability) / math.log(4))

    return entropy


# method to get the unique features in the attribute
def get_values(data, attributes, attribute):

    index = attributes.index(attribute)

    values = []


    for entry in data:
        if entry[index] not in values:
            values.append(entry[index])

    return values


# method to return all the data  from the data file excepting the best attribute
def get_data(data, attributes, best_attribute, value):
    data_values = [[]]
    index = attributes.index(best_attribute)    # finding the index of the best attribute chosen

    for row in data:
        if row[index] == value:
            new_entry = []
            # Add values in data_values except the values in the best attribute
            for i in range(0, len(row)):
                if i != index:
                    new_entry.append(row[i])
            data_values.append(new_entry)

    data_values.remove([])
    return data_values


# method to get all unique classification values in target
def get_target_values(data, attributes, target):
    values = []

    for record in data:
        index_of_target = attributes.index(target)
        value = record[index_of_target]
        values.append(value)

    return values


# method to get the most common classification
def get_majority(data, attributes, target):
    frequency ={}
    index = attributes.index(target)
    for tuple in data:
        if (frequency.has_key(tuple[index])):
            frequency[tuple[index]] += 1
        else:
            frequency[tuple[index]] = 1
    max = 0
    major = ""
    for key in frequency.keys():
        if frequency[key] > max:
            max = frequency[key]
            major = key
    return major


xml_string = ""


def generate_d_tree(data, attributes, target, first_run): # method to build decision tree in xml format
    global xml_string

    # deciding on root node
    if first_run:

        with open('decision_tree.xml', 'w') as append_file:
            append_file.write("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>")
        xml_string = ""

        index = attributes.index(target)   # finding index of target classification
        val_freq = {}
        #length = len(data)
        # to find the unique classifications in the target attribute with their count
        for row in data:
            if row[index] in val_freq:
                    val_freq[row[index]] += 1
            else:
                    val_freq[row[index]] = 1


        # To calculate entropy of the entire data set
        system_entropy = calculate_entropy(data, attributes, target)

        val_freq_str = ""
        for each_val in val_freq:
            val_freq_str = val_freq_str + str(each_val) + ","   # + ":" + str(val_freq[each_val]) + ","
        system_entropy = str(system_entropy)

        new_xml = "<tree classes =\"" + val_freq_str + "\" entropy=\"" + system_entropy+"\">"


        xml_string = xml_string + new_xml

        with open('decision_tree.xml', 'a') as file:
            file.write(xml_string)
        xml_string = ""

    # Get all values of the target attribute (Ex: unacc, unacc, unacc, acc, acc, good)
    values = get_target_values(data, attributes, target)

    # If all rows in the data set have the same classification value in target, return that value
    if values.count(values[0]) == len(values) :
        return values[0]

    # If the data set is empty or the attributes list is empty, return the most common classification
    elif not data or (len(attributes) - 1) <= 0:
        return get_majority(data, attributes, target)

    else:

        first_run = False
        # Choose the next attribute for classification
        best_attribute = select_best_attribute(data, attributes, target)

        values = get_values(data, attributes, best_attribute)  # unique classification values for the best attribute
        for value in values:

            data_for_subtree = get_data(data, attributes, best_attribute, value)

            new_attributes = attributes[:]
            new_attributes.remove(best_attribute)  # eliminating the chosen best attribute

            if not first_run:
                index = new_attributes.index(target)
                val_freq = {}

                for row in data_for_subtree:
                    if row[index] in val_freq:
                        val_freq[row[index]] += 1
                    else:
                        val_freq[row[index]] = 1

                system_entropy = calculate_entropy(data_for_subtree, new_attributes, target)

                val_freq_str = ''
                #feature_str = str("feature=" + best_attribute)
                for each in val_freq:
                    val_freq_str = val_freq_str + str(each) + ","

                    index_best_attribute = attributes.index(best_attribute)
                    feature = attributes[index_best_attribute]
                    feature_str = str('feature=' + '"' + best_attribute + '" ')
                    if system_entropy <= 0.0:
                        temp_str = feature_str + 'value=' + '"' + value + '"' + '>' + each
                    else:
                        temp_str = feature_str + 'value=' + '"' + value + '"' + '>'

                system_entropy = str(system_entropy)

                xml_string = "<node entropy=\"" + system_entropy + "\" " + temp_str

                with open('decision_tree.xml', 'a') as append_file:
                    append_file.write(xml_string)

                xml_string = ''

            generate_d_tree(data_for_subtree, new_attributes, target, first_run)

            with open('decision_tree.xml', 'a') as append_file:
                append_file.write("</node>")


def main():

    data = []  # to store the data in the data file

    print(sys.executable)

    data_file = input('enter the CSV filename : ')

    # reading for attributes in the csv file user entered
    if data_file == 'car.csv':
        with open(data_file, 'r') as file:
            for row in file:
                row = row.strip("\r\n")
                data.append(row.split(','))
        attributes = ['att0', 'att1', 'att2', 'att3', 'att4', 'att5', 'classification']   # describing attributes for car data file
        target = 'classification'

    elif data_file == 'nursery.csv':
        with open(data_file, 'r') as file:
            for row in file:
                if row is None or len(row.strip("\r\n")) == 0:
                    continue
                row = row.strip("\r\n")
                data.append(row.split(','))

        print(row)
        attributes = ['att0', 'att1', 'att2', 'att3', 'att4', 'att5', 'att6', 'att7', 'classification'] # describing attributes for nursery data file
        target = 'classification'
    else:
     print("data file not in records, please choose between 'car' and 'nursery'! ")
     sys.exit()

    first_run = True
    generate_d_tree(data, attributes, target, first_run)

    # Append final XML branch and end the tree growth
    with open('decision_tree.xml', 'a') as append_file:
        append_file.write("</tree>")

    print('Decision Tree generated. Please open "decision_tree.xml" in location.')


main()
