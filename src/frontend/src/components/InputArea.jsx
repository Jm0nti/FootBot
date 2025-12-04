import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const ChatMessage = ({ message }) => {
  const isUser = message.sender === 'user';

  return (
    <div className={`message ${message.type || (isUser ? 'user' : 'bot')}`}>
      {/* Si es usuario, texto plano. Si es bot, Markdown */}
      {isUser ? (
        <p>{message.text}</p>
      ) : (
        <div className="markdown-content">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {message.text}
          </ReactMarkdown>
        </div>
      )}

      {/* Renderizado de imagen si el agente la envía (ej. Formaciones) */}
      {message.image && (
        <img 
          src={`http://localhost:8000${message.image}`} 
          alt="Generated content" 
          className="bot-image" 
        />
      )}
      
      {/* Metadatos técnicos pequeños (opcional) */}
      {!isUser && message.meta && (
        <div style={{fontSize: '0.7rem', color: '#888', marginTop: '5px'}}>
           Agente: {message.meta.agent} | Pasos: {message.meta.interaction_count}
        </div>
      )}
    </div>
  );
};

export default ChatMessage;