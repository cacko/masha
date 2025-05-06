from google.maps import places_v1

from google.oauth2 import service_account


def sample_get_place(query):
    # Create a client
    credentials = service_account.Credentials.from_service_account_file(
        "/Volumes/Devo/Code/alex-1d675-f015eb07a9f7.json"
    )
    client = places_v1.PlacesClient(credentials=credentials)
    fieldMask = "places.formattedAddress,places.displayName"
    # Initialize request argument(s)
    request = places_v1.GetPlaceRequest(
        name=query,
    )

    # Make the request
    response = client.get_place(request=request, metadata=[("x-goog-fieldmask",fieldMask)])

    # Handle the response
    print(response)
