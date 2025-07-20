import React, { useState, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import { FaMicrophone, FaPaperPlane } from 'react-icons/fa';

const App = () => {
  const [universities, setUniversities] = useState([]);
  const [selectedUniversity, setSelectedUniversity] = useState('yeshiva');
  const [input, setInput] = useState('');
  const [chatLog, setChatLog] = useState([
    {
      sender: 'bot',
      text: `Hello! Welcome to YESHIVA UNIVERSITY Bot, how can I assist you today?\n\n**Disclaimer: This chatbot service is not intended for private or confidential information.**`
    }
  ]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    axios.get('/registered-universities')
      .then(res => setUniversities(res.data.universities))
      .catch(err => console.error(err));
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input || !selectedUniversity) return;
    setLoading(true);
    setChatLog(prev => [...prev, { sender: 'user', text: input }]);
    try {
      const res = await axios.post(`/${selectedUniversity}`, { message: input });
      setChatLog(prev => [...prev, { sender: 'bot', text: res.data.response }]);
    } catch (err) {
      console.error(err);
      setChatLog(prev => [...prev, { sender: 'bot', text: 'Error fetching response.' }]);
    }
    setInput('');
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-white flex flex-col">
      {/* Header */}
      <div className="bg-blue-800 text-white text-center py-4 text-3xl font-bold relative">
        {selectedUniversity?.replace(/-/g, ' ').toUpperCase()} Bot
        <button className="absolute top-2 right-4 bg-black text-white px-2 py-1 rounded text-sm">Dark Mode</button>
      </div>

      {/* Main Chat Area */}
      <div className="max-w-3xl w-full mx-auto mt-6 px-4 py-6 bg-gray-100 rounded-3xl flex-1">
        <div className="space-y-4">
          {chatLog.map((msg, idx) => (
            <div key={idx} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[80%] px-4 py-3 rounded-2xl text-sm leading-relaxed shadow-md ${msg.sender === 'user' ? 'bg-blue-100' : 'bg-gray-600 text-white'}`}>
                <ReactMarkdown>{msg.text}</ReactMarkdown>
              </div>
            </div>
          ))}
          {loading && <div className="text-center text-gray-500">Generating response...</div>}
        </div>
      </div>

      {/* Input Area */}
      <form onSubmit={handleSubmit} className="max-w-3xl w-full mx-auto mt-2 flex items-center gap-2 px-4 py-2 border-t">
        <button type="button" className="p-2 text-gray-500">
          <FaMicrophone size={18} />
        </button>
        <input
          type="text"
          className="flex-1 p-2 border rounded-xl"
          placeholder="Type your message here"
          value={input}
          onChange={e => setInput(e.target.value)}
        />
        <button type="submit" className="p-2 text-blue-600">
          <FaPaperPlane size={20} />
        </button>
      </form>

      {/* Footer */}
      <div className="text-xs text-center text-gray-500 mt-2 pb-4">
        This ChatBot can generate inaccurate answers. Check important info.
      </div>
    </div>
  );
};

export default App;
