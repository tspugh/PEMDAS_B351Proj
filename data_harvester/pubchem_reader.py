import requests

#https://github.com/IvanChernyshov/NistChemPy/blob/main/tutorial.ipynb
import nistchempy as nist
from numpy import empty
import os
import pickle

PATH = ".."

# gets every valid molecule from start index to index+amount based on cid on PubChem
# returns an array of length end-start+1 containing every InChI id that is returned
#These are directly used in the get_spectra method to get all the spectra using nistchempy
#
#start - what cid to get from pubchem
#end - what cid to stop at for pubchem
#file_to_write - optional pickle dump
def get_molecule_ids(start, end, file_to_write=None):
    values = empty(end-start+1, dtype="S150")


    for i in range(end-start+1):

        id = i + start

        response = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{id}/property/InChI/TXT")
        if response:
            values[i] = response.text[:-1]
            print("found "+str(i))
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
        for one_id in [x for x in ids if x is not None]:
            search.find_compounds(identifier=one_id, search_type="inchi")
            if search.success and len(search.IDs)>0:
                search.load_found_compounds()
                compound = search.compounds[0]
                print(search.compounds[0])
                compound.get_all_spectra()
                fuldir = dir+"/"+compound.name+"/"
                if not os.path.exists(fuldir):
                    os.makedirs(fuldir)
                compound.save_all_spectra(path_dir=fuldir)
                holder.setdefault(one_id, compound)
    return holder



if __name__ == "__main__":
    #a start and end index of ids to use
    start = 3001
    end = 6000

    #the name of the pickle dump for get_molecule_ids
    id_string = f"tempIds{start},{end}.txt"

    #gets all cids from start-end and writes them to a txt file in pickled form
    file_to_write_to = open(id_string, "wb")
    l = get_molecule_ids(start, end, file_to_write=file_to_write_to)
    file_to_write_to.close()

    #reads from the pickle file and downloads all the spectra
    #file_to_read_from = open(f"tempIds{start},{end}.txt", "rb")
    #l = pickle.load(file_to_read_from)
    #spec = get_spectra(l, start, end)
