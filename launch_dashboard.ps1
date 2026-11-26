param(
    [int]$Port = 8765,
    [switch]$NoBrowser
)

$arguments = @("dashboard\server.py", "--port", $Port)
if ($NoBrowser) { $arguments += "--no-browser" }
python @arguments
