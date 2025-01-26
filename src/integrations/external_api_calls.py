from codecs import oem_decode


def call_osm_api(df, parameters):
    # Example: Add latitude and longitude based on address
    df["latitude"] = df["address"].apply(lambda x: oem_decode(x)["latitude"])
    df["longitude"] = df["address"].apply(lambda x: oem_decode(x)["longitude"])
    return df

import openai

def call_llm_api(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=200
    )
    return response.choices[0].text.strip()
