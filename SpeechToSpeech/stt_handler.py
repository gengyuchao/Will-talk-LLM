import requests
import aiohttp
import asyncio

class InferenceHandler:
    def __init__(self, server_url):
        self.server_url = server_url

    # async def send_inference_request(self, file_path, temperature=0.0, temperature_inc=0.2, response_format='json'):
    #     """Send audio file to inference server and get the text response."""
    #     async with aiohttp.ClientSession() as session:
    #         with open(file_path, 'rb') as file:
    #             data = {
    #                 'temperature': str(temperature),
    #                 'temperature_inc': str(temperature_inc),
    #                 'response_format': response_format
    #             }
    #             files = {'file': file}
    #             async with session.post(self.server_url, data=data, files=files) as response:
    #                 if response.status == 200:
    #                     if response_format == 'json':
    #                         result = await response.json()
    #                         return result['text']
    #                     else:
    #                         return await response.text()
    #                 else:
    #                     raise Exception(f"Failed to get response: {response.status}, {await response.text()}")


    async def send_inference_request(self, file_path, temperature=0.0, temperature_inc=0.2, response_format='json'):
        """Send audio file to inference server and get the text response."""
        async with aiohttp.ClientSession() as session:
            # Create a multipart/form-data payload
            data = aiohttp.FormData()
            data.add_field('file', open(file_path, 'rb'), filename=file_path)
            data.add_field('temperature', str(temperature))
            data.add_field('temperature_inc', str(temperature_inc))
            data.add_field('response_format', response_format)

            async with session.post(self.server_url, data=data) as response:
                if response.status == 200:
                    if response_format == 'json':
                        result = await response.json()
                        return result['text']
                    else:
                        return await response.text()
                else:
                    raise Exception(f"Failed to get response: {response.status}, {await response.text()}")
