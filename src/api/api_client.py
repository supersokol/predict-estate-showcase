# sdk/api_client.py
import requests

class APIClient:
    def __init__(self, base_url: str, token: str = None):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.openapi_spec = self._fetch_openapi_spec()
        self.methods = self._generate_methods()

    def _fetch_openapi_spec(self) -> dict:
        """
        Fetch the OpenAPI specification from the API.
        """
        response = requests.get(f"{self.base_url}/openapi.json")
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception("Failed to fetch OpenAPI specification")

    def _generate_methods(self) -> dict:
        """
        Generate methods dynamically based on OpenAPI specification.
        """
        methods = {}
        for path, endpoints in self.openapi_spec["paths"].items():
            for method, details in endpoints.items():
                func_name = self._generate_function_name(method, path)
                methods[func_name] = self._create_method(func_name, method, path, details)
        return methods

    def _generate_function_name(self, method: str, path: str) -> str:
        """
        Generate a function name based on the HTTP method and path.
        """
        path = path.strip("/").replace("/", "_").replace("{", "").replace("}", "")
        return f"{method.lower()}_{path}"

    def _create_method(self, func_name: str, method: str, path: str, details: dict):
        """
        Create a method for an API endpoint.
        """
        def method_func(**kwargs):
            url = self.base_url + path.format(**kwargs)
            if method.lower() in ["get", "delete"]:
                response = requests.request(method.upper(), url, params=kwargs.get("params"))
            else:
                response = requests.request(method.upper(), url, json=kwargs.get("data"))
            return response.json()

        # Attach metadata to the function
        method_func.metadata = {
            "name": func_name,
            "method": method.upper(),
            "path": path,
            "description": details.get("summary", "No description available"),
            "parameters": details.get("parameters", []),
        }
        return method_func

    def list_methods(self):
        """
        List all dynamically generated methods with their metadata.
        """
        return {name: func.metadata for name, func in self.methods.items()}

    def __getattr__(self, name):
        """
        Dynamically access methods by name.
        """
        if name in self.methods:
            return self.methods[name]
        raise AttributeError(f"Method {name} not found in API client")