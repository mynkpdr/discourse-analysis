import requests
import os

cookies = {
    '_fbp': 'fb.2.1730635235140.34945671877610009',
    '_ga_QDQG66T9X3': 'GS2.1.s1756552132$o1$g1$t1756552196$j60$l0$h20506298',
    'forced_color_mode': 'dark',
    '_ga_QHXRKWW9HH': 'GS2.3.s1768143967$o6$g0$t1768143967$j60$l0$h0',
    '_ga': 'GA1.1.1135173043.1730635235',
    '_gcl_au': '1.1.1953537757.1770444868',
    '_ga_YRLBGM14X9': 'GS2.1.s1770638273$o1$g1$t1770638494$j60$l0$h0',
    '_ga_5HTJMW67XK': 'GS2.1.s1770959028$o391$g0$t1770959045$j43$l0$h0',
    '_ga_08NPRH5L4M': 'GS2.1.s1770996932$o912$g1$t1770997032$j60$l0$h0',
    '_forum_session': 'm0Q4JfViuU9Eakr%2BwnENX9Hv58iVulVeUqVNgI1ROwJYfGogaLeaKefbUW0rDkm5rLt%2F%2BtLrTXD0Zqem%2F3MGi7ip8D0GK9uLestq5cSdVJvA0igDuqSjNh5VX0zcAqnOi2qO%2Bn2H10ZwvH7nkf%2BsHwh4I8f%2BYzUwkzPpTwVpcfMsFvdSqLhQsJ3quY%2BtNntaYtp57mAXvWIDW51aqt1B3jrxXeDUlBkFFiCtCTG4Vu0bJJcO1rqKcMEjOf9TeHlo4WItU93vLMnQzcw4Tns%3D--klOjftnDETh17i4v--viQepBF7KlsX4Q0h811Kxw%3D%3D',
    '_t': '%2BcHjG9Ta5Ri%2BuX8vH22XKYSTWvXvGLX4EjmEd8zFKVgf%2Feyvj7GTRTyvRMmzc3cqvhY%2Bw0LhwlyG1hgl9YchsAxzA8g46cifp8Ww27r6i3y%2BHcIJtO55%2Ft3ylPVuDGabHHhlCRQF1U1SiffuwIgvN6uAApa%2BbjpFDaXSzax5Rha8neNlgNxLUo0bv%2FrzlPAA54YYi8GKSoAJHCueeNO8nCVHkAelaEe8%2B3tOwmg%2BmNsSWUQtD1NWJMdL6YdChNfdRjZmM35WlpgvrErda%2FOo6PjigRY9jKhaPZ9VzjAEVZs%3D--0EPsRcwH5CujvPv7--Epvftwnd0UYXvA5mH6sCbA%3D%3D',
}

headers = {
    'accept': 'application/json, text/javascript, */*; q=0.01',
    'accept-language': 'en-US,en;q=0.9',
    'cache-control': 'no-cache',
    'discourse-logged-in': 'true',
    'discourse-present': 'true',
    'pragma': 'no-cache',
    'priority': 'u=1, i',
    'referer': 'https://discourse.onlinedegree.iitm.ac.in/u?order=likes_received&period=all',
    'sec-ch-ua': '"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36',
    'x-csrf-token': 'vHRWOG3a9YcJX87IXZS8yF2GJ44qmNqCTvlxjL4Pu2S3KcIzmLtL6kLlURQ_NxDO-RR0hjDpUJmPRXWJ0Z1ZAw',
    'x-requested-with': 'XMLHttpRequest',
    # 'cookie': '_fbp=fb.2.1730635235140.34945671877610009; _ga_QDQG66T9X3=GS2.1.s1756552132$o1$g1$t1756552196$j60$l0$h20506298; forced_color_mode=dark; _ga_QHXRKWW9HH=GS2.3.s1768143967$o6$g0$t1768143967$j60$l0$h0; _ga=GA1.1.1135173043.1730635235; _gcl_au=1.1.1953537757.1770444868; _ga_YRLBGM14X9=GS2.1.s1770638273$o1$g1$t1770638494$j60$l0$h0; _ga_5HTJMW67XK=GS2.1.s1770959028$o391$g0$t1770959045$j43$l0$h0; _ga_08NPRH5L4M=GS2.1.s1770996932$o912$g1$t1770997032$j60$l0$h0; _forum_session=m0Q4JfViuU9Eakr%2BwnENX9Hv58iVulVeUqVNgI1ROwJYfGogaLeaKefbUW0rDkm5rLt%2F%2BtLrTXD0Zqem%2F3MGi7ip8D0GK9uLestq5cSdVJvA0igDuqSjNh5VX0zcAqnOi2qO%2Bn2H10ZwvH7nkf%2BsHwh4I8f%2BYzUwkzPpTwVpcfMsFvdSqLhQsJ3quY%2BtNntaYtp57mAXvWIDW51aqt1B3jrxXeDUlBkFFiCtCTG4Vu0bJJcO1rqKcMEjOf9TeHlo4WItU93vLMnQzcw4Tns%3D--klOjftnDETh17i4v--viQepBF7KlsX4Q0h811Kxw%3D%3D; _t=%2BcHjG9Ta5Ri%2BuX8vH22XKYSTWvXvGLX4EjmEd8zFKVgf%2Feyvj7GTRTyvRMmzc3cqvhY%2Bw0LhwlyG1hgl9YchsAxzA8g46cifp8Ww27r6i3y%2BHcIJtO55%2Ft3ylPVuDGabHHhlCRQF1U1SiffuwIgvN6uAApa%2BbjpFDaXSzax5Rha8neNlgNxLUo0bv%2FrzlPAA54YYi8GKSoAJHCueeNO8nCVHkAelaEe8%2B3tOwmg%2BmNsSWUQtD1NWJMdL6YdChNfdRjZmM35WlpgvrErda%2FOo6PjigRY9jKhaPZ9VzjAEVZs%3D--0EPsRcwH5CujvPv7--Epvftwnd0UYXvA5mH6sCbA%3D%3D',
}

os.makedirs("data", exist_ok=True)

for page in range(1, 11):
    params = {
        "period": "all",
        "order": "likes_received",
        "page": str(page),
    }

    response = requests.get(
        "https://discourse.onlinedegree.iitm.ac.in/directory_items",
        params=params,
        cookies=cookies,
        headers=headers,
    )

    data = response.json()

    with open(f"data/page_{page}.json", "w", encoding="utf-8") as f:
        import json
        json.dump(data, f, indent=2)

    print(f"Saved page {page}")
