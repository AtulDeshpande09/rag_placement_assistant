import requests

def get_page(url):

    try:
        response = requests.get(url)

        if response.status_code == 200:
            html_content = response.text

            with open(f"../data/{url}.html", 'w') as f:
                f.write(html_content)
        
    
    except request.exceptions.RequestException as e:
        print(f"An error occurred : {e}")


def save_pages(links):

    for i in links:
        get_page(i)

    print("saved pages")
