import requests

#https://github.com/IvanChernyshov/NistChemPy/blob/main/tutorial.ipynb
import nistchempy as nist
from numpy import empty
import os
import pickle
from time import sleep

PATH = ".."
HEAVY_ATOM_CUTOFF = 30
MUST_HAVE_NITROGEN = True

def meets_criteria(molecule_name:str):
    try:
        response = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"
            f"/{molecule_name}/property/InChI,CanonicalSMILES,Title,"
            f"HeavyAtomCount/JSON")
        if response:
            obj_gotten = response.json()["PropertyTable"][
                              "Properties"][0]
            print(obj_gotten)
            if obj_gotten is None:
                return [False, None]
            return json_meets_criteria(obj_gotten), obj_gotten
        return [False, None]
    except Exception:
        print("error was made")
        sleep(1)
        return [False, None]


def json_meets_criteria(mol_json):
    keys = mol_json.keys()
    if "HeavyAtomCount" in keys and "CanonicalSMILES" in keys and \
            "Title" in keys:
        if  ("Ni" in mol_json[
        "CanonicalSMILES"]) or ("Na" in mol_json[
        "CanonicalSMILES"]) or ("Nb" in mol_json[
        "CanonicalSMILES"]) or ("Ne" in mol_json[
        "CanonicalSMILES"]) or ("Mn" in mol_json[
        "CanonicalSMILES"]) or ("Rn" in mol_json[
        "CanonicalSMILES"]) or ("No" in mol_json[
        "CanonicalSMILES"]) or ("Zn" in mol_json[
        "CanonicalSMILES"]) or ("Sn" in mol_json[
        "CanonicalSMILES"]):
            print("shoot")
            return False
        return (int(mol_json["HeavyAtomCount"]) <= HEAVY_ATOM_CUTOFF)
    return False

# gets every valid molecule from start index to index+amount based on cid on PubChem
# returns an array of length end-start+1 containing every InChI id that is returned
#These are directly used in the get_spectra method to get all the spectra using nistchempy
#
#start - what cid to get from pubchem
#end - what cid to stop at for pubchem
#file_to_write - optional pickle dump
def get_molecule_ids(start, end, file_to_write=None):
    values = []


    for i in range(end-start+1):

        id = i + start

        try:
            response = requests.get(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid"
                f"/{id}/property/InChI,CanonicalSMILES,Title,"
                f"HeavyAtomCount/JSON")
            if response:
                j = response.json()["PropertyTable"][
                                  "Properties"][0]
                values.append(j)
                print("found "+str(i))
            else:
                values.append(None)
        except Exception:
            print("error for now")
            sleep(1)
    if file_to_write is not None:
        pickle.dump(values, file_to_write, protocol=pickle.HIGHEST_PROTOCOL)
    return values



# returns a single JCAMP-DX file for a spectrum of type "type" and InChI id "id"
# ex: get_spectra(["oeijewjriej", "InChI=a/eiowCh;lieri"], "IR")
#
# returns a JCAMP-DX file's output
#
#ids - list of all ids to read from
#start - the start index used in get_molecule_ids
#end - the end index used in get_molecule_ids
def get_spectra(ids, start, end):

    dir = os.path.join(PATH, "spectra", f"{start}-{end}")
    if not os.path.exists(dir):
        os.makedirs(dir)

    holder = dict()
    if ids is not None:
        search = nist.Search(NoIon=True, cMS=True)
        for i in range(len(ids)):
            one_id = ids[i]["InChI"]
            if one_id is not None:
                search.find_compounds(identifier=one_id, search_type="inchi")
                if search.success and len(search.IDs)>0:
                    search.load_found_compounds()
                    compound = search.compounds[0]
                    print(search.compounds[0])
                    compound.get_all_spectra()
                    nitrogen = False
                    if "N" in ids[i][
                            "CanonicalSMILES"]:
                        nitrogen = True
                        fuldir = dir+"/Nitrogenic/"+ids[i]["Title"]+"/"
                    else:
                        fuldir = dir + "/NoNitrogen/" + ids[i][
                            "Title"] + "/"
                    if not os.path.exists(fuldir):
                        os.makedirs(fuldir)
                    q = open(os.path.join(fuldir, "classification_info.txt"),"w")
                    q.write(
                        ids[i]["Title"] + "\n" + ids[i][
                            "CanonicalSMILES"] + "\n" +
                             f""
                             f"{'Positive' if nitrogen else 'Negative'}")
                    q.close()
                    compound.save_all_spectra(path_dir=fuldir)
                    holder.setdefault(one_id, compound)
                else:
                    ids[i] = None

    return holder, refine(ids)

def refine(dicts):

    final = []
    for i in dicts:
        if i is not None and json_meets_criteria(i):
            final.append(i)

    return final

#takes in a list of dictionaries as returned from
# either get_molecule_ids or refine, and writes in a form suitable to
# be read later
def dump_smiles(dictionary, file_to_dump_to):
    for x in dictionary:
        file_to_dump_to.write(x["CanonicalSMILES"]+"~"+x["Title"]+"\n")

# a start and end index of ids to use
def run_spectra_and_save(start, end, load_from_file=False):

    # the name of the pickle dump for get_molecule_ids
    id_string = f"tempIds{start},{end}.txt"

    # gets all cids from start-end and writes them to a txt file in pickled form
    if not load_from_file:
        file_to_write_to = open(id_string, "wb")
        l = get_molecule_ids(start, end, file_to_write=file_to_write_to)
        file_to_write_to.close()
    else:
        # reads from the pickle file and downloads all the spectra
        file_to_read_from = open(f"tempIds{start},{end}.txt", "rb")
        l = pickle.load(file_to_read_from)

    # ensures all values meet requirements of our project for now
    refined_molecules = refine(l)

    spec, refined_molecules = get_spectra(refined_molecules, start, end)

    id_string_smile = f"smiles{start},{end}.txt"
    f = open(id_string_smile, "w")
    dump_smiles(refined_molecules, f)
    f.close()

if __name__ == "__main__":
    run_spectra_and_save(150001,300000,True)
