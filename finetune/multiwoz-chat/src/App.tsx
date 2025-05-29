import React, { useRef, useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";

type Message = { role: "user" | "agent"; content: string };

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  // Scroll to bottom on new message
  const scrollRef = useRef<HTMLDivElement>(null);
  React.useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollIntoView({ behavior: "smooth" });
  }, [messages, streaming]);

  const sendMessage = async () => {
    if (!input.trim() || streaming) return;
    setMessages([...messages, { role: "user", content: input }]);
    setStreaming(true);

    // Connect WebSocket
    const ws = new WebSocket("ws://localhost:8000/ws");
    wsRef.current = ws;
    let agentMsg = "";

    ws.onopen = () => {
      // Send only the new user message
      ws.send(JSON.stringify({ user_message: input }));
      setInput("");
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.token) {
        agentMsg += data.token;
        setMessages((prev) => [
          ...prev.filter((_, idx) => idx !== prev.length), // Remove last temp agent msg
          ...[],
          ...(agentMsg
            ? [{ role: "agent", content: agentMsg }]
            : [])
        ]);
      } else if (data.done) {
        setStreaming(false);
        setMessages((prev) => [
          ...prev.filter((m, i) => i !== prev.length - 1),
          { role: "agent", content: data.reply }
        ]);
        ws.close();
      }
    };

    ws.onerror = (e) => {
      setStreaming(false);
      ws.close();
      alert("WebSocket error. Is the backend running?");
    };
    ws.onclose = () => setStreaming(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") sendMessage();
  };

  return (
    <div className="bg-background min-h-screen flex items-center justify-center p-4">
      <Card className="w-full max-w-xl shadow-2xl rounded-2xl">
        <CardContent className="p-6 flex flex-col h-[75vh]">
          <h1 className="text-2xl font-bold mb-4">MultiWOZ Travel Chatbot</h1>
          <ScrollArea className="flex-1 mb-4 pr-2 overflow-y-auto">
            <div>
              {messages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`mb-2 flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`px-4 py-2 rounded-2xl text-base max-w-xs whitespace-pre-line ${
                      msg.role === "user"
                        ? "bg-blue-500 text-white"
                        : "bg-gray-200 text-gray-900"
                    }`}
                  >
                    {msg.content}
                  </div>
                </div>
              ))}
              {streaming && (
                <div className="flex justify-start">
                  <div className="px-4 py-2 rounded-2xl bg-gray-200 text-gray-900 animate-pulse">
                    ...
                  </div>
                </div>
              )}
              <div ref={scrollRef} />
            </div>
          </ScrollArea>
          <div className="flex gap-2 mt-auto">
            <Input
              placeholder="Ask about your travel, e.g. Book a train to Cambridge"
              className="flex-1"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={streaming}
            />
            <Button onClick={sendMessage} disabled={streaming || !input.trim()}>
              Send
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
