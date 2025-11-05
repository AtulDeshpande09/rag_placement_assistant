import requests
from get_links import search_web ,get_links

def get_page(url,num):

    try:
        response = requests.get(url)

        if response.status_code == 200:
            html_content = response.text

            with open(f"../data/{num}.md", 'w') as f:
                f.write(html_content)
        
    
    except requests.exceptions.RequestException as e:
        print(f"An error occurred : {e}")


def save_pages(links):

    for i,link in enumerate(links):
        get_page(link,i)

    print("saved pages")


if __name__ == '__main__':

    query = "Interview questions for python developer"

    result = search_web(query)
    
    data = result["organic_results"]
    links = get_links(data)

    save_pages(links)
