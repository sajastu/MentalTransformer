import json

ents = []
counter = 0
with open('/disk1/sajad/gov-reports/test.json') as fR:
    for l in fR:
        ents.append(json.loads(l.strip()))
        counter += 1
        if counter == 10:
            break


with open('/disk1/sajad/gov-reports/test-sample.json', mode='w') as fW:
    for e in ents:
        json.dump(e, fW)
        fW.write('\n')