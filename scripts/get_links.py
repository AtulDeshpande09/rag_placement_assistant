from serpapi imort GoogleSearch

def search_web(query,location="India",num=10):

    params = {
            'q':query,
            'location': location,
            'google_domain' : 'google.com',
            'num': num,
            'output' : json,
            'api_key': "API_KEY"
            }

    search = GoogleSearch(params):
    results = search.get_dict()
    return results


def get_links(page:dict):
    
    links = []


    return links



def combine(query):

    page = search_web()
    links = get_links(page)

    return links
