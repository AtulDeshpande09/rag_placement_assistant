from serpapi import GoogleSearch
import json

def search_web(query,location="India",num=10):

    params = {
            'q':query,
            'location': location,
            'google_domain' : 'google.com',
            'num': num,
            'output' : json,
            'api_key': "API_KEY"
            }

    search = GoogleSearch(params)
    results = search.get_dict()
    return results


def get_links(results:dict):
    
    links = []
    for result in results:
        link = result["link"]
        links.append(link)

    return links



def combine(query):

    page = search_web()
    page = page["organic_results"]
    links = get_links(page)

    return link


if __name__ == '__main__':

    query = "Interview questions for python developer"

    result = search_web(query)
    
    data = result["organic_results"]
    link = get_links(data)
    print(link)
