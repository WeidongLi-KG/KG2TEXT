from collections import Counter, OrderedDict
import pickle
import json
import argparse
import pandas as pd


class Read_file:
    """Read table and description files"""
    def __init__(self, num_sample=None, min_freq_fields=100, type=0, max_len=100, maxp=0):
        prefix = "../Wiki_dataset/"
        self.type = type
        stats = []
        for mode in range(3):
            self.mode = mode
            if mode == 0:
                path = "train_"
            else:
                num_sample /= 8
                if self.type == 0:
                    path = "../train_P.pkl"
                else:
                    path = "../train_A.pkl"
                with open(path, 'rb') as output:
                    data = pickle.load(output)
                maxp = data["maxp"]
                if mode == 1:
                    path = "valid_"
                else:
                    path = "test_"
            if type == 0:
                post = "wiki_P.json"
            else:
                post = "wiki_A.json"
            table_path = prefix + path + post
            self.maxp = maxp

            self.num_sample = num_sample
            self.max_len = max_len
            # least common fields
            self.min_freq_fields = min_freq_fields
            self.sources, self.targets, self.chooses = self.prepare(table_path)
            print('num entities: %d'%len(self.sources))
            stats.append(len(self.sources))
            print("Finish read text")
            self._dump(mode)
            print("Finish dump")
        pd.DataFrame([stats], columns=['train', 'valid', 'test']).to_csv('stats.csv')

    def prepare(self, path):
        field_corpus = []
        old_targets = []
        old_table = []
        old_chooses = []
        with open(path, 'r') as files:
            i = 0
            for line in files:
                temp_table = json.loads(line.strip('\n'))
                table, target, flag, field, choose = self.turnc_sent(temp_table)
                if flag:
                    old_targets.append(target)
                    old_chooses.append(choose)
                    old_table.append({key: value for key, value in table.items() if key != "TEXT"})
                    field_corpus.extend(field)
                    i += 1
                    if i == self.num_sample * 1.5:
                        break
        fields = Counter(field_corpus)
        if self.min_freq_fields:
            fields = {word: freq for word, freq in fields.items() if freq >= self.min_freq_fields}
        used_field = list(fields)
        sources = []
        targets = []
        chooses = []
        j = 0
        for i, table in enumerate(old_table):
            keys = [key for key in table.keys() if key in used_field and key != "Name_ID"]
            index = 1
            temp = [("Name_ID", table["Name_ID"], index)]
            index += 1
            order_values = []
            for key in keys:
                for item in table[key]:
                    if item["mainsnak"] not in order_values:
                        temp.append((key, item["mainsnak"], index))
                        order_values.append(item["mainsnak"])
                        if "qualifiers" in item:
                            qualifiers = item['qualifiers']
                            for qkey, qitems in qualifiers.items():
                                if qkey in used_field:
                                    for qitem in qitems:
                                        if qitems not in order_values:
                                            temp.append((qkey, qitem, index))
                                            order_values.append(qitems)
                        index += 1
            if self.maxp < index:
                if self.mode == 0:
                    self.maxp = index
                else:
                    continue
            if self.type == 0 and len(temp) < 5:
                continue
            else:
                if len(temp) < 3:
                    continue
            new_sent = []
            new_choose = []
            for sent, cho in zip(old_targets[i], old_chooses[i]):
                for word in order_values:
                    if word in sent:
                        new_sent.extend(sent)
                        new_choose.extend(cho)
                        break
            if len(new_sent) < 5:
                continue
            if not self.filt_data(temp, new_sent):
                continue
            sources.append(temp)
            j += 1
            targets.append(new_sent)
            chooses.append(new_choose)

            if j == self.num_sample:
                break
        # pprint(sources)
        return sources, targets, chooses

    def filt_data(self, source, sentence):
        values = []; used_values = []

        for item in source:
            values.append(item[1])
        for i in sentence:
            if i in values:
                used_values.append(i)
        if len(values) == len(used_values):
            return True
        return False

    def ranksent(self, order_values, target):
        final_target = []; choose = []
        target_dict = {}
        for j, sent in enumerate(target):
            tmp = []
            for word in order_values:
                try:
                    i = sent.index(word)
                except:
                    pass
                else:
                    tmp.append(i)
            if len(tmp) > 0:
                target_dict[j] = min(tmp)
        for index in sorted(target_dict, key=target_dict.get):
            final_target.append(target[index])
        for sent in final_target:
            choose.append([0 if value in order_values else 1 for value in sent])
        return final_target, choose

    def turnc_sent(self, table):
        values = set()
        order_values = [table["Name_ID"]]
        for key, items in table.items():
            if key == "Name_ID" or key == "TEXT":
                continue
            for item in items:
                values.add(item["mainsnak"])
                if item["mainsnak"] not in order_values and key != "given name":
                    order_values.append(item["mainsnak"])
                if "qualifiers" in item:
                    qualifiers = item['qualifiers']
                    for _, qitems in qualifiers.items():
                        values.update(qitems)
                        for qitem in qitems:
                            if qitem not in order_values:
                                order_values.append(qitems)
        target = table["TEXT"]
        used_value = set()
        final_sent = []
        final_target = []
        final_choose = []
        target, chooses = self.ranksent(order_values, target)
        for sent, choose in zip(target, chooses):
            if len(final_target) + len(sent) > self.max_len:
                break
            else:
                final_sent.append(sent)
                final_choose.append(choose)
                final_target.extend(sent)
        for word in final_target:
            if word in values:
                used_value.add(word)
        if self.type == 0 and len(used_value) < 5:
            return None, None, False, None, None
        newinfobox = OrderedDict()
        update_value = set()
        field = []
        for key, items in table.items():
            if key == "Name_ID":
                field.append(key)
                newinfobox["Name_ID"] = items
                continue
            if key == "TEXT":
                continue
            item_list = []
            for item in items:
                new_value = {}
                if item['mainsnak'] in used_value:
                    new_value['mainsnak'] = item['mainsnak']
                    update_value.add(item['mainsnak'])
                    field.append(key)
                    if 'qualifiers' in item:
                        qualifiers = item['qualifiers']
                        new_qualifer = OrderedDict()
                        for qkey, qitems in qualifiers.items():
                            qitem_list = []
                            for qitem in qitems:
                                if qitem in used_value:
                                    field.append(qkey)
                                    qitem_list.append(qitem)
                                    update_value.add(qitem)
                            if len(qitem_list) > 0:
                                new_qualifer[qkey] = qitem_list
                        if len(new_qualifer) > 0:
                            new_value['qualifiers'] = new_qualifer
                    if len(new_value) > 0:
                        item_list.append(new_value)
            if len(item_list) > 0:
                newinfobox[key] = item_list
        if self.type == 0 and len(update_value) < 5:
            return None, None, False, None, None
        return newinfobox, final_sent, True, field, final_choose

    def _dump(self, mode):
        print(len(self.sources))
        if mode == 0:
            if self.type == 0:
                path = "../train_P.pkl"
            else:
                path = "../train_A.pkl"
        elif mode == 1:
            if self.type == 0:
                path = "../valid_P.pkl"
            else:
                path = "../valid_A.pkl"
        else:
            if self.type == 0:
                path = "../test_P.pkl"
            else:
                path = "../test_A.pkl"
        data = {
            "source": self.sources,
            "target": self.targets,
            "choose": self.chooses,
            "maxp": self.maxp
        }
        with open(path, 'wb') as output:
            pickle.dump(data, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess')
    parser.add_argument('--type', type=int,  default=0,
                        help='per(0)/other(1)')
    args = parser.parse_args()
    Read_file(type=args.type, num_sample=500000)