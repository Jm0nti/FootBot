import { useState, useRef, useEffect } from 'react';
import ChatMessage from './components/ChatMessage';
import './App.css';

function App() {
  const [messages, setMessages] = useState([
    { 
      text: "¡Hola! Soy tu asistente experto en fútbol. Pregúntame sobre estadísticas, historias de clubes o reglas. 📊⏰📋", 
      sender: 'bot',
      meta: { agent: 'System', interaction_count: 0 }
    }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Auto-scroll al fondo
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { text: input, sender: 'user' };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage.text })
      });

      if (!response.ok) throw new Error('Error en el servidor');

      const data = await response.json();
      
      // Construimos el mensaje del bot con la data del backend
      const botMessage = {
        text: data.answer,
        sender: 'bot',
        image: data.image, // URL de la imagen si existe
        meta: data.meta    // Info de trazabilidad
      };

      setMessages((prev) => [...prev, botMessage]);

    } catch (error) {
      setMessages((prev) => [...prev, { 
        text: "Error: No pude conectar con el agente. Verifica que server.py esté corriendo.", 
        sender: 'bot', 
        type: 'error' 
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async () => {
    if(!confirm("¿Borrar historial?")) return;
    try {
      await fetch('http://localhost:8000/api/reset', { method: 'POST' });
      setMessages([{ text: "Memoria reiniciada. ¿En qué puedo ayudarte ahora?", sender: 'bot' }]);
    } catch (error) {
      console.error("Error reseteando", error);
    }
  };

  return (
    <>
      <header>
        <div className="header-title">
          <img src="/soccer-ball.png" alt="Soccer Ball" className="title-icon" />
          <h1>FootBot</h1>
        </div>
        <div className="header-actions">
          <img src="/whistle.png" alt="Whistle" className="whistle-icon" />
          <button onClick={handleReset} className="reset-btn" disabled={loading}>
            Reiniciar Chat
          </button>
        </div>
      </header>

      <div className="chat-container">
        {messages.map((msg, idx) => (
          <ChatMessage key={idx} message={msg} />
        ))}
        {loading && (
          <div className="message bot">
            <p>Pensando...</p>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form className="input-area" onSubmit={handleSend}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Escribe tu consulta de fútbol aquí..."
          disabled={loading}
          autoFocus
        />
        <button type="submit" className="send-btn" disabled={loading || !input.trim()} title="Enviar">
          {loading ? '...' : 'Enviar'}
        </button>
      </form>
    </>
  );
}

export default App;