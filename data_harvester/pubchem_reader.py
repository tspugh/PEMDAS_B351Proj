
import requests
import nistchempy as nist
from numpy import empty


_search = None
#gets every valid molecule from start index to index+amount based on cid on PubChem
#returns an array of length amount containing every InChI id that is returned
def get_molecule_ids(start, amount):
    values = empty(amount, dtype="S150")

    for i in range(amount):

        id = i+start

        response = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{id}/property/InChI/TXT")
        if response:
            values[i] = response.text[:-1]

    return values

def initialize_search():
    _search = nist.Search(NoIon=True, cMS=True)


#returns a single JCAMP-DX file for a spectrum of type "type" and InChI id "id"
#ex: get_spectrum("oeijewjriej", "IR")
def get_spectrum(id: str, type: str):
    if id is not None:
        _search.find_compoounds(identifier = id, search_type = "InChI")
        if _search.success:
            _search.load_found_compounds()
            compound = _search.compounds[0]
            if compound.get_spectra(type) is not None:
                spectrum = compound.get_spectra(type)[0]
    return None


if __name__ == "__main__":

    print(get_molecule_ids(0,5))