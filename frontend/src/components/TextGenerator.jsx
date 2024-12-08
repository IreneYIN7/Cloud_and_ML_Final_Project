import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Slider } from '@/components/ui/slider';
import { Loader2 } from 'lucide-react';

const TextGenerator = () => {
  const [input, setInput] = useState('');
  const [output, setOutput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [maxLength, setMaxLength] = useState(100);

  const generateText = async () => {
    setLoading(true);
    setError('');
    
    try {
      const response = await fetch('http://localhost:8000/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: input,
          max_length: maxLength
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to generate text');
      }

      const data = await response.json();
      setOutput(data.generated_text);
    } catch (err) {
      setError(err.message || 'Error generating text. Please try again.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="w-full max-w-2xl mx-auto mt-8">
      <CardHeader>
        <CardTitle>Text Generator</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-2">
          <label className="text-sm font-medium">Input Text</label>
          <Textarea
            placeholder="Enter your text here..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            className="min-h-32"
          />
        </div>
        
        <div className="space-y-2">
          <label className="text-sm font-medium">Maximum Length: {maxLength}</label>
          <Slider
            value={[maxLength]}
            onValueChange={(value) => setMaxLength(value[0])}
            min={50}
            max={200}
            step={10}
            className="w-full"
          />
        </div>

        <Button 
          onClick={generateText}
          disabled={loading || !input.trim()}
          className="w-full"
        >
          {loading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Generating...
            </>
          ) : (
            'Generate Text'
          )}
        </Button>

        {error && (
          <div className="p-4 bg-red-50 text-red-500 rounded-md text-sm">
            {error}
          </div>
        )}

        {output && (
          <div className="space-y-2">
            <label className="text-sm font-medium">Generated Text:</label>
            <div className="p-4 bg-gray-50 rounded-md whitespace-pre-wrap">
              {output}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default TextGenerator;